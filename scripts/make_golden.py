"""Generate the golden-vector porting contract: fixed-seed input→output MIDI pairs.

The vectors under tests/golden/ are the executable byte-level contract for the engine
(see wobblemidi_determinism.md and wobblemidi_porting_contract.md): 10 input fixtures
and ~26 (input, params) → output pairs sweeping the full parameter surface, verified
on every test run by tests/test_golden_vectors.py / scripts/verify_golden.py.

REGENERATION IS A DELIBERATE ACT. The checked-in outputs define current engine
behaviour; regenerating them redefines the contract. Only do it alongside an
intentional engine/profile behaviour change, in the same commit, and say so in the
commit message. This script refuses to overwrite an existing manifest without --force.

Usage:
    python scripts/make_golden.py            # first generation (no manifest present)
    python scripts/make_golden.py --force    # deliberate regeneration
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version as pkg_version
from importlib.resources import as_file, files
from pathlib import Path

import click
import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))      # sibling scripts (make_demo, make_eartest)

import make_demo                                     # noqa: E402
import make_eartest                                  # noqa: E402

from wobblemidi.humanise import humanise, load_profile   # noqa: E402

GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
PPQ = 480
SIXTEENTH = PPQ // 4          # 120
EIGHTH = PPQ // 2             # 240
TEMPO_120 = 500_000
KICK, SNARE, HAT_CLOSED, CRASH = 36, 38, 42, 49

# The full humanise() parameter surface, pinned explicitly per vector so the contract
# never silently moves with a product-default change (defaults are product decisions;
# the vectors lock ENGINE behaviour). The one exception is f1_cli below, which runs
# through the CLI with only --seed and therefore deliberately locks the CLI defaults.
ENGINE_PARAMS_PINNED = {
    "genre": "rock",
    "beat_type": "beat",
    "intensity": 0.35,
    "seed": 42,
    "timing_only": False,
    "velocity_only": False,
    "push": False,
    "phi": 0.4,
    "all_channels": False,
    "push_amount": None,
    "intensity_by_group": None,
}

# (vector_id, input_fixture_stem, param overrides | {"cli_args": [...]})
VECTORS: list[tuple[str, str, dict]] = [
    # Core surface: every fixture at pinned defaults.
    ("f1_default", "f1_rock_4bar", {}),
    ("f2_default", "f2_rock_ghosts", {}),
    ("f3_default", "f3_four_floor", {}),
    ("f4_default", "f4_flam_cluster", {}),
    ("f5_default", "f5_sixeight", {}),
    ("f6_default", "f6_threefour", {}),
    ("f7_default", "f7_multitrack", {}),
    ("f8_default", "f8_tempomap", {}),
    ("f9_empty_default", "f9_empty", {}),
    ("f9_single_default", "f9_single_hit", {}),
    # Parameter sweep on the canonical beat.
    ("f1_intensity00", "f1_rock_4bar", {"intensity": 0.0}),
    ("f1_intensity10", "f1_rock_4bar", {"intensity": 1.0}),
    ("f1_phi00", "f1_rock_4bar", {"phi": 0.0}),
    ("f1_phi09", "f1_rock_4bar", {"phi": 0.9}),
    ("f1_push", "f1_rock_4bar", {"push": True}),
    ("f1_lean_mirror", "f1_rock_4bar", {"push_amount": -1.0}),
    ("f1_lean_half", "f1_rock_4bar", {"push_amount": 0.5}),
    ("f1_seed7", "f1_rock_4bar", {"seed": 7}),
    # Coupling-window fixture: coupling disabled (phi=0 smears the flams — locked as
    # the documented pre-coupling behaviour), and cluster min-eff with a pinned lane.
    ("f4_phi00", "f4_flam_cluster", {"phi": 0.0}),
    ("f4_hats_pinned", "f4_flam_cluster", {"intensity_by_group": {"hihat_closed": 0.0}}),
    # Mode flags + section on the tier-exercising ghosts pattern.
    ("f2_timing_only", "f2_rock_ghosts", {"timing_only": True}),
    ("f2_velocity_only", "f2_rock_ghosts", {"velocity_only": True}),
    ("f2_fill", "f2_rock_ghosts", {"beat_type": "fill"}),
    # Multi-track: per-lane gains and the drum-channel filter opt-out.
    ("f7_lanes", "f7_multitrack", {"intensity_by_group": {"snare": 0.0, "hihat_closed": 1.0}}),
    ("f7_all_channels", "f7_multitrack", {"all_channels": True}),
    # Flag plumbing: through the click CLI (in-process), locking CLI defaults too.
    ("f1_cli", "f1_rock_4bar", {"cli_args": ["--seed", "42"]}),
]

# Vector pairs that must produce DIFFERENT bytes (seed sensitivity).
SEED_SENSITIVITY_PAIRS = [("f1_default", "f1_seed7")]


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _track(note_events, tempo_events=(), timesig=(4, 4)) -> MidiTrack:
    """Build one MidiTrack. note_events: (tick, channel, note, velocity, dur_ticks).

    Same-tick ordering matches the existing builders: tempo metas, then note_offs,
    then note_ons (stable).
    """
    seq: list[tuple[int, int, mido.Message | mido.MetaMessage]] = []
    for t, tempo in tempo_events:
        seq.append((t, 0, MetaMessage("set_tempo", tempo=tempo, time=0)))
    for t, ch, note, vel, dur in note_events:
        seq.append((t, 2, Message("note_on", channel=ch, note=note, velocity=vel, time=0)))
        seq.append((t + dur, 1, Message("note_off", channel=ch, note=note, velocity=0, time=0)))
    seq.sort(key=lambda e: (e[0], e[1]))
    tr = MidiTrack()
    if timesig is not None:
        tr.append(MetaMessage("time_signature", numerator=timesig[0], denominator=timesig[1],
                              clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    prev = 0
    for t, _, msg in seq:
        tr.append(msg.copy(time=t - prev))
        prev = t
    tr.append(MetaMessage("end_of_track", time=0))
    return tr


def _save(path: Path, tracks: list[MidiTrack], midi_type: int) -> None:
    mid = MidiFile(type=midi_type, ticks_per_beat=PPQ)
    mid.tracks.extend(tracks)
    mid.save(str(path))


def _fx_flam_cluster(path: Path) -> None:
    """Coupling-window exerciser: flams inside COUPLE_WINDOW_MS, same-tick chords,
    a 1-tick hat inside a cluster (elastic member offs), and a fixed melodic wall
    just after a flam (cluster-scope clamp). 2 bars, 120 BPM → 1 tick ≈ 1.042 ms,
    so the 8-tick grace gap ≈ 8.3 ms sits inside the 12 ms window."""
    ev: list[tuple[int, int, int, int, int]] = []
    for bar in (0, 1):
        b = bar * 4 * PPQ
        for i in range(8):                      # straight-8th hats
            t = b + i * EIGHTH
            dur = 1 if (bar == 0 and i == 4) else 60   # 1-tick hat inside the beat-3 cluster
            ev.append((t, 9, HAT_CLOSED, 84 if i % 2 == 0 else 68, dur))
        ev.append((b, 9, KICK, 105, 60))                     # beat 1
        ev.append((b + 2 * PPQ, 9, KICK, 110, 60))           # beat 3
        ev.append((b + PPQ - 8, 9, SNARE, 40, 30))           # flam grace, 8 ticks early
        ev.append((b + PPQ, 9, SNARE, 112, 60))              # flam main on beat 2
        ev.append((b + 3 * PPQ, 9, SNARE, 108, 60))          # backbeat 4
    ev.append((0, 9, CRASH, 100, 120))                       # same-tick chord on the downbeat
    ev.append((2 * PPQ + 8, 9, SNARE, 42, 30))               # ruff after bar-1 beat 3 (with 1-tick hat)
    ev.append((4 * PPQ + PPQ + 12, 0, 60, 80, 100))          # melodic ch-0 wall after bar-2 flam
    _save(path, [_track(ev, tempo_events=[(0, TEMPO_120)])], midi_type=0)


def _fx_sixeight(path: Path) -> None:
    """Uniform 6/8 (eighth-note grid path, grid_pos=None). 4 bars."""
    ev = []
    for bar in range(4):
        b = bar * 6 * EIGHTH
        ev.append((b, 9, KICK, 112, 60))
        ev.append((b + 3 * EIGHTH, 9, SNARE, 106, 60))
        for i in range(6):
            ev.append((b + i * EIGHTH, 9, HAT_CLOSED, 80 if i in (0, 3) else 64, 40))
    _save(path, [_track(ev, tempo_events=[(0, TEMPO_120)], timesig=(6, 8))], midi_type=0)


def _fx_threefour(path: Path) -> None:
    """Uniform 3/4 (16th grid, positional lookups off: grid_pos=None). 4 bars."""
    ev = []
    for bar in range(4):
        b = bar * 3 * PPQ
        ev.append((b, 9, KICK, 112, 60))
        ev.append((b + 2 * PPQ, 9, SNARE, 106, 60))
        for i in range(6):
            ev.append((b + i * EIGHTH, 9, HAT_CLOSED, 82 if i % 2 == 0 else 66, 40))
    _save(path, [_track(ev, tempo_events=[(0, TEMPO_120)], timesig=(3, 4))], midi_type=0)


def _fx_multitrack(path: Path) -> None:
    """Type 1: kick+snare track, hat track (same-tick cross-track chords), and a
    melodic ch-0 bass track on drum-range note numbers (channel-filter fixture). 4 bars."""
    t_kick_snare = []
    t_hats = []
    t_bass = []
    for bar in range(4):
        b = bar * 4 * PPQ
        t_kick_snare.append((b, 9, KICK, 108, 60))
        t_kick_snare.append((b + 2 * PPQ, 9, KICK, 108, 60))
        t_kick_snare.append((b + PPQ, 9, SNARE, 106, 60))
        t_kick_snare.append((b + 3 * PPQ, 9, SNARE, 106, 60))
        for i in range(8):
            t_hats.append((b + i * EIGHTH, 9, HAT_CLOSED, 86 if i % 2 == 0 else 72, 40))
        t_bass.append((b, 0, 36, 96, 2 * PPQ - 80))          # drum-range notes, melodic channel
        t_bass.append((b + 2 * PPQ, 0, 38, 92, 2 * PPQ - 80))
    _save(path, [
        _track(t_kick_snare, tempo_events=[(0, TEMPO_120)]),
        _track(t_hats, timesig=None),
        _track(t_bass, timesig=None),
    ], midi_type=1)


def _fx_tempomap(path: Path) -> None:
    """Mid-file tempo changes, including one at tick 3660 — BETWEEN the 16th grid
    ticks 3600 and 3720 — so offset application walks a tempo boundary. 4 bars."""
    ev = []
    for bar in range(4):
        b = bar * 4 * PPQ
        ev.append((b, 9, KICK, 108, 60))
        ev.append((b + 2 * PPQ, 9, KICK, 104, 60))
        ev.append((b + PPQ, 9, SNARE, 106, 60))
        ev.append((b + 3 * PPQ, 9, SNARE, 106, 60))
        for i in range(8):
            ev.append((b + i * EIGHTH, 9, HAT_CLOSED, 85 if i % 2 == 0 else 70, 40))
    tempos = [(0, TEMPO_120), (4 * PPQ, 666_667), (3660, 545_455), (12 * PPQ, TEMPO_120)]
    _save(path, [_track(ev, tempo_events=tempos)], midi_type=0)


def _fx_empty(path: Path) -> None:
    """No note events at all — metas only."""
    _save(path, [_track([], tempo_events=[(0, TEMPO_120)])], midi_type=0)


def _fx_single_hit(path: Path) -> None:
    """One kick at tick 0: locks the tick-0 lower-bound clamp on a negative offset."""
    _save(path, [_track([(0, 9, KICK, 100, 60)], tempo_events=[(0, TEMPO_120)])], midi_type=0)


def build_fixtures(inputs_dir: Path) -> None:
    make_demo.write_midi(inputs_dir / "f1_rock_4bar.mid", make_demo.build_beat())
    make_eartest.build_input(make_eartest.PATTERNS["rock_ghosts"]).save(
        str(inputs_dir / "f2_rock_ghosts.mid"))
    make_eartest.build_input(make_eartest.PATTERNS["four_floor"]).save(
        str(inputs_dir / "f3_four_floor.mid"))
    _fx_flam_cluster(inputs_dir / "f4_flam_cluster.mid")
    _fx_sixeight(inputs_dir / "f5_sixeight.mid")
    _fx_threefour(inputs_dir / "f6_threefour.mid")
    _fx_multitrack(inputs_dir / "f7_multitrack.mid")
    _fx_tempomap(inputs_dir / "f8_tempomap.mid")
    _fx_empty(inputs_dir / "f9_empty.mid")
    _fx_single_hit(inputs_dir / "f9_single_hit.mid")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _run_vector(vector_id: str, input_path: Path, output_path: Path,
                overrides: dict, profile) -> dict:
    if "cli_args" in overrides:
        from click.testing import CliRunner
        from wobblemidi.cli import main as cli_main
        result = CliRunner().invoke(
            cli_main, [str(input_path), str(output_path), *overrides["cli_args"]])
        if result.exit_code != 0:
            raise RuntimeError(f"{vector_id}: CLI failed ({result.exit_code}): {result.output}")
        return {"cli_args": overrides["cli_args"]}
    params = {**ENGINE_PARAMS_PINNED, **overrides}
    humanise(input_path, output_path, profile, **params)
    return {"params": params}


@click.command()
@click.option("--force", is_flag=True,
              help="Overwrite an existing manifest — a DELIBERATE contract regeneration.")
def main(force: bool) -> None:
    """Generate tests/golden/: fixtures, vector outputs, and the manifest."""
    manifest_path = GOLDEN_DIR / "manifest.json"
    if manifest_path.exists() and not force:
        raise click.ClickException(
            "tests/golden/manifest.json already exists. Regenerating golden vectors "
            "REDEFINES the engine's byte-level contract — only do this alongside an "
            "intentional behaviour change, in the same commit. Re-run with --force."
        )

    inputs_dir = GOLDEN_DIR / "inputs"
    outputs_dir = GOLDEN_DIR / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    build_fixtures(inputs_dir)
    click.echo(f"fixtures: {len(list(inputs_dir.glob('*.mid')))} files in {inputs_dir}")

    profile_res = files("wobblemidi.profiles").joinpath("rock.json")
    with as_file(profile_res) as p:
        profile_sha = _sha256(Path(p))
        profile = load_profile(p)

    entries = []
    for vector_id, stem, overrides in VECTORS:
        input_path = inputs_dir / f"{stem}.mid"
        output_path = outputs_dir / f"{vector_id}.mid"
        spec = _run_vector(vector_id, input_path, output_path, overrides, profile)
        entries.append({
            "id": vector_id,
            "input": f"inputs/{stem}.mid",
            "output": f"outputs/{vector_id}.mid",
            **spec,
            "sha256_input": _sha256(input_path),
            "sha256_output": _sha256(output_path),
        })
        click.echo(f"  {vector_id}")

    manifest = {
        "_meta": {
            "description": (
                "Golden vectors: fixed-seed input->output pairs locking engine behaviour "
                "byte-for-byte. Verify: scripts/verify_golden.py (or pytest "
                "tests/test_golden_vectors.py). Regenerate ONLY deliberately: "
                "scripts/make_golden.py --force, committed together with the engine/"
                "profile change that motivated it. See wobblemidi_determinism.md."
            ),
            "generated_at_rev": _git_rev(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": pkg_version("numpy"),
            "scipy": pkg_version("scipy"),
            "mido": pkg_version("mido"),
            "profile": "wobblemidi/profiles/rock.json",
            "profile_sha256": profile_sha,
            "seed_sensitivity_pairs": SEED_SENSITIVITY_PAIRS,
        },
        "vectors": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    click.echo(f"manifest: {manifest_path} ({len(entries)} vectors)")
    click.echo("\nNOTE: these files define the engine's byte-level contract. Commit them "
               "and never regenerate casually.")


if __name__ == "__main__":
    main()
