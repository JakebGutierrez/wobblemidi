"""Generate the module-13 velocity-rebuild A/B ear test.

Writes two programmed 8-bar rock patterns — "rock_ghosts" (snare ghosts + busy
three-level 16th hi-hats, the cases the velocity rebuild targets) and
"four_floor" (basic four-on-the-floor, the plainest groove test) — then
humanises each twice with the SAME seed and default settings, once per profile,
so the results can be A/B'd in a DAW:

    demo/eartest/{pattern}_input.mid   the programmed pattern (on-grid palette)
    demo/eartest/{pattern}_old.mid     --old profile ("before")
    demo/eartest/{pattern}_new.mid     --candidate profile ("after")

To A/B against a historical profile, extract it from git first, e.g.:
    git show <commit>:pocketmidi/profiles/rock.json > demo/eartest/before.json
    python scripts/make_eartest.py --old demo/eartest/before.json

What to listen for: with the old profile, snare ghosts jump loud / backbeats duck
(accent structure sampled as noise) and the hats machine-gun between levels; with
the rebuilt profile the programmed dynamics survive with human-scale variation.

Usage:
    python scripts/make_eartest.py [--candidate validation/candidate_new_schema.json]
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import mido

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pocketmidi.humanise import humanise, load_profile

PPQ = 480
SIXTEENTH = PPQ // 4
NOTE_DUR = SIXTEENTH // 2      # half a 16th: realistic programmed length, ample window
TEMPO_US = 631_579             # ~95 BPM — pocket rock, ghosts clearly audible
BARS = 8
OUT_DIR = REPO_ROOT / "demo" / "eartest"

KICK, SNARE, HAT_CLOSED, HAT_OPEN, CRASH = 36, 38, 42, 46, 49

# One bar on the 16th grid (positions 0-15). (position, note, velocity, bar_filter)
# bar_filter: None = every bar, else a predicate on the 0-based bar index.
GHOSTS_PATTERN: list[tuple[int, int, int, object]] = [
    # kick — two-level pattern, syncopated push into beat 3
    (0,  KICK, 112, None),
    (6,  KICK, 96,  None),
    (8,  KICK, 108, None),
    (14, KICK, 94,  lambda bar: bar % 4 == 3),          # fill-in kick every 4th bar
    # snare — backbeats + classic ghost placements (e of 3, a of 3, a of 4)
    (4,  SNARE, 106, None),
    (12, SNARE, 106, None),
    (7,  SNARE, 26,  None),                             # ghost
    (10, SNARE, 28,  None),                             # ghost
    (15, SNARE, 27,  lambda bar: bar % 2 == 1),         # ghost, alternate bars
    # hats — busy 16ths, three-level accent contour; open hat lifts bar 4/8
    *[(p, HAT_CLOSED, 96 if p % 4 == 0 else (78 if p % 2 == 0 else 62), None)
      for p in range(16) if p != 14],
    (14, HAT_CLOSED, 78, lambda bar: bar % 4 != 3),
    (14, HAT_OPEN, 92, lambda bar: bar % 4 == 3),
    # crash on the very first downbeat only
    (0, CRASH, 110, lambda bar: bar == 0),
]

# Basic four-on-the-floor rock beat: kick on every quarter, backbeat snare,
# straight accented 8th hats. No ghosts — the plainest possible groove test.
FOUR_FLOOR_PATTERN: list[tuple[int, int, int, object]] = [
    (0,  KICK, 112, None), (4, KICK, 112, None), (8, KICK, 112, None), (12, KICK, 112, None),
    (4,  SNARE, 106, None), (12, SNARE, 106, None),
    *[(p, HAT_CLOSED, 92 if p % 4 == 0 else 70, None) for p in range(0, 16, 2)],
    (0, CRASH, 110, lambda bar: bar == 0),
]

PATTERNS = {"rock_ghosts": GHOSTS_PATTERN, "four_floor": FOUR_FLOOR_PATTERN}


def build_input(pattern: list) -> mido.MidiFile:
    events = []   # (tick, off_first_priority, msg)
    for bar in range(BARS):
        bar_tick = bar * 16 * SIXTEENTH
        for pos, note, vel, cond in pattern:
            if cond is not None and not cond(bar):
                continue
            on = bar_tick + pos * SIXTEENTH
            events.append((on, 1, mido.Message(
                "note_on", channel=9, note=note, velocity=vel, time=0)))
            events.append((on + NOTE_DUR, 0, mido.Message(
                "note_off", channel=9, note=note, velocity=0, time=0)))
    events.sort(key=lambda e: (e[0], e[1]))

    mid = mido.MidiFile(type=0, ticks_per_beat=PPQ)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO_US, time=0))
    prev = 0
    for tick, _, msg in events:
        track.append(msg.copy(time=tick - prev))
        prev = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(track)
    return mid


@click.command()
@click.option("--old", "old_profile", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=REPO_ROOT / "pocketmidi" / "profiles" / "rock.json", show_default=True,
              help="The 'before' profile.")
@click.option("--candidate", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=REPO_ROOT / "validation" / "candidate_new_schema.json", show_default=True,
              help="The 'after' (rebuilt) profile.")
@click.option("--seed", default=42, show_default=True,
              help="Shared seed: both renders draw from identical RNG streams.")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path),
              default=OUT_DIR, show_default=True)
@click.option("--timing-sweep", default="0.3,0.5,0.7", show_default=True,
              help="Comma-separated intensities for extra timing-only renders "
                   "(velocities untouched); empty string to skip.")
def main(old_profile: Path, candidate: Path, seed: int, out_dir: Path,
         timing_sweep: str) -> None:
    """Write input/old/new ear-test files for the velocity-rebuild A/B."""
    timing_sweep = [float(s) for s in timing_sweep.split(",") if s.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)

    old_prof = load_profile(old_profile)
    cand_prof = load_profile(candidate)

    for pname, pattern in PATTERNS.items():
        input_path = out_dir / f"{pname}_input.mid"
        build_input(pattern).save(str(input_path))
        click.echo(f"{pname} input:  {input_path}")
        for label, prof, prof_path in (("old", old_prof, old_profile),
                                       ("new", cand_prof, candidate)):
            out_path = out_dir / f"{pname}_{label}.mid"
            humanise(input_path, out_path, prof,
                     genre="rock", beat_type="beat", seed=seed)
            click.echo(f"{pname} {label}:    {out_path}   (profile: {prof_path})")

    # Diagnostic legs (rock_ghosts only): each humanisation axis in isolation,
    # candidate profile only. The RNG streams are isolated by design, so with the
    # same seed these decompose the full render exactly: new_velonly carries
    # new.mid's velocities on the input's grid timing, and new_timingonly carries
    # new.mid's timing with the input's programmed velocities.
    input_path = out_dir / "rock_ghosts_input.mid"
    for label, kwargs in (("new_velonly", {"velocity_only": True}),
                          ("new_timingonly", {"timing_only": True})):
        out_path = out_dir / f"rock_ghosts_{label}.mid"
        humanise(input_path, out_path, cand_prof,
                 genre="rock", beat_type="beat", seed=seed, **kwargs)
        click.echo(f"{label.replace('_', ' '):<15}: {out_path}")

    # Timing-only intensity sweep at EXPLICIT intensities: same seed → identical
    # offset draws, scaled linearly toward the grid. Isolates "too much timing
    # spread" from "spread feels random regardless of amount" (which would point
    # at the correlation structure, not the amount). The un-suffixed legs above
    # render at the humanise() default intensity.
    for i in timing_sweep:
        out_path = out_dir / f"rock_ghosts_new_timingonly_i{int(round(i * 100)):02d}.mid"
        humanise(input_path, out_path, cand_prof,
                 genre="rock", beat_type="beat", seed=seed,
                 timing_only=True, intensity=i)
        click.echo(f"timing i={i:.1f}  : {out_path}")

    import inspect
    default_intensity = inspect.signature(humanise).parameters["intensity"].default
    click.echo(f"\nSame seed, default settings (intensity {default_intensity}, phi 0.4, no push).")
    click.echo("Listen for: ghost/backbeat roles surviving on the snare, and the")
    click.echo("hi-hat accent contour staying intact instead of machine-gunning.")


if __name__ == "__main__":
    main()
