"""Generate the committed release demo pack: before/after MIDI pairs in demo/.

Renders the two make_eartest.py patterns (rock_ghosts, four_floor) plus a flam
pattern with the SHIPPED rock profile at the engine's default settings and a
pinned seed, so every pair is reproducible byte-for-byte:

    demo/rock_ghosts_input.mid / demo/rock_ghosts_humanised.mid
    demo/four_floor_input.mid  / demo/four_floor_humanised.mid
    demo/flam_beat_input.mid   / demo/flam_beat_humanised.mid
    demo/flam_beat_uncoupled.mid   (same seed, --groove-tightness 0 — coupling off)

The phi A/B trio (demo/rock_4bar_*.mid) is generated separately by
scripts/make_demo.py; demo/README.md documents both commands.

    python scripts/make_demo_pack.py            # defaults: seed 42, ./demo
"""

from __future__ import annotations

import argparse
import sys
from importlib.resources import as_file, files
from pathlib import Path

import mido

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from make_eartest import PATTERNS, build_input  # noqa: E402
from wobblemidi.humanise import humanise, load_profile  # noqa: E402

SEED = 42

# --- flam pattern: 4 bars, 120 BPM, snare flams on beat 4 of bars 2 & 4 -------
PPQ = 480
SIXTEENTH = PPQ // 4
BAR = 16 * SIXTEENTH
TEMPO_US = 500_000          # 120 BPM → 1 tick ≈ 1.042 ms at PPQ 480
KICK, SNARE, HAT_CLOSED = 36, 38, 42
FLAM_GRACE_TICKS = 8        # ≈ 8.3 ms before the main stroke — inside COUPLE_WINDOW_MS
NOTE_DUR = 60
GRACE_DUR = 4               # grace note_off must land before the main stroke's note_on


def build_flam_beat() -> list[tuple[int, int, int, int]]:
    """Return (abs_tick, note, velocity, duration) for a rock beat with snare flams."""
    events: list[tuple[int, int, int, int]] = []
    for bar in range(4):
        base = bar * BAR
        for pos in (0, 8):
            events.append((base + pos * SIXTEENTH, KICK, 108, NOTE_DUR))
        for pos in (4, 12):
            main = base + pos * SIXTEENTH
            if pos == 12 and bar in (1, 3):      # flam: grace stroke just ahead
                events.append((main - FLAM_GRACE_TICKS, SNARE, 45, GRACE_DUR))
            events.append((main, SNARE, 106, NOTE_DUR))
        for pos in range(0, 16, 2):
            vel = 85 if pos in (0, 4, 8, 12) else 72
            events.append((base + pos * SIXTEENTH, HAT_CLOSED, vel, NOTE_DUR))
    return events


def write_midi(path: Path, events: list[tuple[int, int, int, int]]) -> None:
    mid = mido.MidiFile(type=0, ticks_per_beat=PPQ)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=TEMPO_US, time=0))
    tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4,
                               clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    seq: list[tuple[int, int, int, int]] = []
    for t, note, vel, dur in events:
        seq.append((t, 1, note, vel))
        seq.append((t + dur, 0, note, 0))
    seq.sort(key=lambda e: (e[0], e[1]))         # at a tick: offs before ons; stable
    prev = 0
    for t, kind, note, vel in seq:
        tr.append(mido.Message("note_on", channel=9, note=note,
                               velocity=vel if kind else 0, time=t - prev))
        prev = t
    tr.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED, help="RNG seed (default 42).")
    ap.add_argument("--outdir", default=REPO_ROOT / "demo", type=Path,
                    help="Output directory (default ./demo).")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with as_file(files("wobblemidi.profiles").joinpath("rock.json")) as p:
        prof = load_profile(p)

    # Ear-test patterns, engine defaults (intensity/phi come from humanise()).
    for name, pattern in PATTERNS.items():
        src = args.outdir / f"{name}_input.mid"
        build_input(pattern).save(str(src))
        out = args.outdir / f"{name}_humanised.mid"
        humanise(src, out, prof, genre="rock", beat_type="beat", seed=args.seed)
        print(f"{src}\n{out}")

    # Flam pattern: default render + coupling-off comparison at the same seed.
    src = args.outdir / "flam_beat_input.mid"
    write_midi(src, build_flam_beat())
    out = args.outdir / "flam_beat_humanised.mid"
    humanise(src, out, prof, genre="rock", beat_type="beat", seed=args.seed)
    uncoupled = args.outdir / "flam_beat_uncoupled.mid"
    humanise(src, uncoupled, prof, genre="rock", beat_type="beat", seed=args.seed, phi=0.0)
    print(f"{src}\n{out}\n{uncoupled}")


if __name__ == "__main__":
    main()
