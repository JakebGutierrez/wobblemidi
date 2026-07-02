"""Estimate a data-grounded --groove-tightness (phi) from the Groove MIDI Dataset.

The groove-drift clock is AR(1); the *applied* output's lag-1 timing autocorrelation is
``(1 - RESIDUAL_SHARE) * phi`` (the independent residual dilutes it). So we measure the
lag-1 autocorrelation ``r`` of successive per-grid-slot timing deviations in GMD rock and
invert:  ``phi = r / (1 - RESIDUAL_SHARE)``.

Read-only: prints a recommendation. Does NOT modify rock.json or the CLI default — adopting
the number is a taste call (A/B it against 0.5 by ear).

Method (matches build_profiles.py's offset math):
  * rock, 4/4 only; ghost kick/snare (< VELOCITY_FLOOR) dropped, same as profile building.
  * per hit: signed offset_ms from its quantised 16th grid slot (tempo-map aware).
  * de-lean: subtract the pooled per-(instrument, grid_pos) mean, so the systematic lean
    (handled statically in humanise) doesn't masquerade as clock correlation.
  * collapse chords: one deviation per occupied grid slot (mean), since coupled same-tick
    hits share the nudge and would otherwise inflate r.
  * pool consecutive (slot_t, slot_{t+1}) pairs within each track; r = Pearson over the pool.

Usage:  python scripts/calibrate_phi.py <path/to/groove>
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import click
import mido
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pocketmidi.humanise import RESIDUAL_SHARE
from pocketmidi.midi_utils import (
    TD11_TO_GROUP,
    build_tempo_map,
    grid_position_in_bar,
    is_four_four,
    quantise_to_grid,
    ticks_to_ms_with_map,
)

VELOCITY_FLOOR = 20
KICK_SNARE = {"kick", "snare"}
MIN_TRACK_HITS = 8


def _signed_offset_ms(abs_tick: int, grid_tick: int, tempo_map, ppq: int) -> float:
    if abs_tick >= grid_tick:
        return ticks_to_ms_with_map(grid_tick, abs_tick, tempo_map, ppq)
    return -ticks_to_ms_with_map(abs_tick, grid_tick, tempo_map, ppq)


def _collect(gmd_dir: Path):
    """Return (tracks, n_rock, files_skipped). Each track = {'hits': [...], 'bpm': float},
    hits time-ordered dicts: grid_tick, grid_pos, group, offset_ms."""
    info = pd.read_csv(gmd_dir / "info.csv")
    rock = info[info["style"].str.startswith("rock")]
    tracks: list[dict] = []
    skipped = 0
    for row in rock.itertuples():
        p = gmd_dir / row.midi_filename
        if not p.exists():
            skipped += 1
            continue
        try:
            midi = mido.MidiFile(str(p))
        except Exception:
            skipped += 1
            continue
        if not is_four_four(midi):
            skipped += 1
            continue
        ppq = midi.ticks_per_beat
        tmap = build_tempo_map(midi)
        for track in midi.tracks:
            hits = []
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type != "note_on" or msg.velocity == 0:
                    continue
                if msg.note not in TD11_TO_GROUP:
                    continue
                grp = TD11_TO_GROUP[msg.note]
                if grp in KICK_SNARE and msg.velocity < VELOCITY_FLOOR:
                    continue
                gt = quantise_to_grid(abs_tick, ppq)
                hits.append({
                    "grid_tick": gt,
                    "grid_pos": grid_position_in_bar(gt, ppq),
                    "group": grp,
                    "offset_ms": _signed_offset_ms(abs_tick, gt, tmap, ppq),
                })
            if len(hits) >= MIN_TRACK_HITS:
                tracks.append({"hits": hits, "bpm": float(row.bpm)})
    return tracks, len(rock), skipped


def _lag1_r(tracks, delean: bool = True, collapse: bool = True):
    """Pooled lag-1 autocorrelation of per-slot timing deviations. Returns (r, n_pairs)."""
    lean: dict = {}
    if delean:
        acc = defaultdict(list)
        for tr in tracks:
            for h in tr["hits"]:
                acc[(h["group"], h["grid_pos"])].append(h["offset_ms"])
        lean = {k: float(np.mean(v)) for k, v in acc.items()}

    def resid(h):
        return h["offset_ms"] - (lean.get((h["group"], h["grid_pos"]), 0.0) if delean else 0.0)

    xs, ys = [], []
    for tr in tracks:
        if collapse:
            by_slot = defaultdict(list)
            for h in tr["hits"]:
                by_slot[h["grid_tick"]].append(resid(h))
            seq = [float(np.mean(by_slot[k])) for k in sorted(by_slot)]
        else:
            seq = [resid(h) for h in tr["hits"]]  # already time-ordered
        xs.extend(seq[:-1])
        ys.extend(seq[1:])
    if len(xs) < 100:
        return float("nan"), len(xs)
    return float(np.corrcoef(xs, ys)[0, 1]), len(xs)


def _phi(r: float) -> float:
    return max(0.0, min(0.99, r / (1.0 - RESIDUAL_SHARE)))


@click.command()
@click.argument("gmd_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def main(gmd_dir: Path) -> None:
    tracks, n_rock, skipped = _collect(gmd_dir)
    total_hits = sum(len(t["hits"]) for t in tracks)
    click.echo(f"Rock files: {n_rock}  (skipped non-4/4/unreadable: {skipped})")
    click.echo(f"Usable tracks: {len(tracks)}   total hits: {total_hits}")

    r, npairs = _lag1_r(tracks, delean=True, collapse=True)
    r_raw, _ = _lag1_r(tracks, delean=False, collapse=False)
    r_coll, _ = _lag1_r(tracks, delean=False, collapse=True)

    click.echo("")
    click.echo(f"lag-1 autocorrelation r (de-leaned + chords collapsed): {r:+.3f}  [{npairs} pairs]")
    click.echo(f"  reference — raw (no de-lean, no collapse):            {r_raw:+.3f}")
    click.echo(f"  reference — collapsed only (no de-lean):             {r_coll:+.3f}")
    click.echo("")
    click.echo(f"RESIDUAL_SHARE (beta) = {RESIDUAL_SHARE}")
    click.echo(f"==> recommended phi = r/(1-beta) = {_phi(r):.3f}   (current default: 0.5)")

    click.echo("")
    click.echo("By tempo (de-leaned + collapsed):")
    for lo, hi, label in [(0, 100, "<100 bpm"), (100, 130, "100-130 bpm"), (130, 10_000, ">=130 bpm")]:
        sub = [t for t in tracks if lo <= t["bpm"] < hi]
        if sub:
            rr, npp = _lag1_r(sub, delean=True, collapse=True)
            click.echo(f"  {label:12s}: r={rr:+.3f}  phi={_phi(rr):.3f}  ({len(sub)} tracks, {npp} pairs)")


if __name__ == "__main__":
    main()
