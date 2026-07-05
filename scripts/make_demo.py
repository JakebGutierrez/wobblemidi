"""Generate an A/B groove-drift demo for listening tests.

Builds a canonical 4-bar 4/4 rock beat (dead on the grid), then humanises it twice with the
SAME seed — phi=0.0 (old feel: every hit timed independently) and phi=0.5 (groove drift +
coupled hits) — using the shipped rock profile. Drag the two outputs into Logic to compare.

    python scripts/make_demo.py                 # defaults: seed 7, intensity 0.85, ./demo
    python scripts/make_demo.py --intensity 1.0 --seed 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from importlib.resources import as_file, files

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

from wobblemidi.humanise import humanise, load_profile

PPQ = 480
SIXTEENTH = PPQ // 4      # 120 ticks
TEMPO_US = 500_000        # 120 BPM
BAR = 16 * SIXTEENTH      # 1920 ticks

# Roland TD-11 notes
KICK, SNARE, HAT_CLOSED, CRASH = 36, 38, 42, 49


def build_beat() -> list[tuple[int, int, int]]:
    """Return (abs_tick, note, velocity) for a straight 4-bar rock beat, quantised dead."""
    hat_accent = {0, 4, 8, 12}
    events: list[tuple[int, int, int]] = []
    for bar in range(4):
        base = bar * BAR
        for pos in (0, 8):                       # kick on beats 1 & 3
            events.append((base + pos * SIXTEENTH, KICK, 108))
        for pos in (4, 12):                      # snare backbeat on 2 & 4
            events.append((base + pos * SIXTEENTH, SNARE, 106))
        if bar in (1, 3):                        # ghost snare (& of 3) in bars 2 & 4
            events.append((base + 10 * SIXTEENTH, SNARE, 30))
        for pos in range(0, 16, 2):              # straight-8th closed hats
            events.append((base + pos * SIXTEENTH, HAT_CLOSED, 85 if pos in hat_accent else 72))
        if bar == 0:                             # crash on the downbeat of bar 1
            events.append((base, CRASH, 112))
    return events


def write_midi(path: Path, events: list[tuple[int, int, int]], dur: int = 60) -> None:
    mid = MidiFile(type=0, ticks_per_beat=PPQ)
    tr = MidiTrack()
    mid.tracks.append(tr)
    tr.append(MetaMessage("set_tempo", tempo=TEMPO_US, time=0))
    tr.append(MetaMessage("time_signature", numerator=4, denominator=4,
                          clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    seq: list[tuple[int, int, int, int]] = []
    for t, note, vel in events:
        seq.append((t, 1, note, vel))            # note_on
        seq.append((t + dur, 0, note, 0))        # note_off
    seq.sort(key=lambda e: (e[0], e[1]))         # at a tick: offs (0) before ons (1); stable
    prev = 0
    for t, kind, note, vel in seq:
        tr.append(Message("note_on", channel=9, note=note,
                          velocity=vel if kind else 0, time=t - prev))
        prev = t
    tr.append(MetaMessage("end_of_track", time=0))
    mid.save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intensity", type=float, default=0.85,
                    help="Humanisation strength 0.0-1.0 (default 0.85 — high, so the phi "
                         "difference is clearly audible).")
    ap.add_argument("--seed", type=int, default=7, help="Shared RNG seed (default 7).")
    ap.add_argument("--outdir", default="demo", help="Output directory (default ./demo).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    src = outdir / "rock_4bar_input.mid"
    write_midi(src, build_beat())

    with as_file(files("wobblemidi.profiles").joinpath("rock.json")) as p:
        prof = load_profile(p)

    phi0 = outdir / "rock_4bar_phi0.mid"
    phi05 = outdir / "rock_4bar_phi05.mid"
    humanise(src, phi0, prof, intensity=args.intensity, seed=args.seed, phi=0.0)
    humanise(src, phi05, prof, intensity=args.intensity, seed=args.seed, phi=0.5)

    print(f"seed={args.seed}  intensity={args.intensity}")
    print(f"  {src}          (programmed, on the grid)")
    print(f"  {phi0}   (phi=0.0 — old feel, independent per-hit timing)")
    print(f"  {phi05}  (phi=0.5 — groove drift + coupled hits)")


if __name__ == "__main__":
    main()
