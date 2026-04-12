"""Build timing/velocity profiles from the Groove MIDI Dataset (GMD).

Usage:
    python scripts/build_profiles.py <path/to/groove-v1.0.0>

Output:
    pocketmidi/profiles/rock.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import click
import mido
import numpy as np
import pandas as pd

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from pocketmidi.midi_utils import (
    TD11_TO_GROUP,
    build_tempo_map,
    quantise_to_grid,
    ticks_to_ms_with_map,
)

MIN_SAMPLES = 30
VELOCITY_FLOOR = 20
KICK_SNARE_GROUPS = {"kick", "snare"}
STRATIFIED_GROUPS = {"kick", "snare"}  # only these get velocity tier buckets
KDE_BW_METHOD = "scott"  # change to "silverman" or a float if hi-hat sounds smeared
OUTPUT_FILE = Path(__file__).parent.parent / "pocketmidi" / "profiles" / "rock.json"


def _velocity_tier(velocity: float, thresholds: tuple[float, float]) -> str:
    low, high = thresholds
    if velocity < low:
        return "soft"
    elif velocity < high:
        return "medium"
    else:
        return "hard"


def _build_pairs(hits: list[dict]) -> list[list[float]]:
    """Convert a list of raw hit dicts to [[offset_ms, vel_delta], ...] pairs."""
    velocities = [h["velocity"] for h in hits]
    median_vel = float(np.median(velocities))
    return [[h["offset_ms"], h["velocity"] - median_vel] for h in hits]


@click.command()
@click.argument("gmd_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def main(gmd_dir: Path) -> None:
    """Ingest GMD rock files and write timing/velocity profiles to pocketmidi/profiles/rock.json."""
    # ------------------------------------------------------------------
    # 1. Load metadata and filter to rock files only
    # ------------------------------------------------------------------
    info = pd.read_csv(gmd_dir / "info.csv")
    rock = info[info["style"].str.startswith("rock")]
    click.echo(f"Rock files: {len(rock)}")

    # ------------------------------------------------------------------
    # 2. Collect raw hits from every rock MIDI file
    # ------------------------------------------------------------------
    raw_hits: list[dict] = []
    skipped_files = 0

    for row in rock.itertuples():
        midi_path = gmd_dir / row.midi_filename
        if not midi_path.exists():
            skipped_files += 1
            continue

        try:
            midi = mido.MidiFile(str(midi_path))
        except Exception:
            skipped_files += 1
            continue

        ppq = midi.ticks_per_beat
        tempo_map = build_tempo_map(midi)

        for track in midi.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time  # delta → absolute, per-track

                if msg.type != "note_on" or msg.velocity == 0:
                    continue
                if msg.note not in TD11_TO_GROUP:
                    continue

                instrument_group = TD11_TO_GROUP[msg.note]

                # Ghost note filter — kick and snare only
                if instrument_group in KICK_SNARE_GROUPS and msg.velocity < VELOCITY_FLOOR:
                    continue

                grid_tick = quantise_to_grid(abs_tick, ppq)

                # Signed offset: positive = late, negative = early
                # Use ticks_to_ms_with_map for both legs so tempo changes
                # between the note and its grid point are handled correctly.
                if abs_tick >= grid_tick:
                    offset_ms = ticks_to_ms_with_map(grid_tick, abs_tick, tempo_map, ppq)
                else:
                    offset_ms = -ticks_to_ms_with_map(abs_tick, grid_tick, tempo_map, ppq)

                raw_hits.append(
                    {
                        "beat_type": row.beat_type,
                        "instrument_group": instrument_group,
                        "offset_ms": offset_ms,
                        "velocity": float(msg.velocity),
                    }
                )

    click.echo(f"Total raw hits collected: {len(raw_hits)}  (files skipped: {skipped_files})")

    # ------------------------------------------------------------------
    # 3. Compute velocity tertile thresholds for kick and snare
    #    (from post-filter raw_hits so boundaries match the retained data)
    # ------------------------------------------------------------------
    all_by_instrument: dict[str, list[float]] = defaultdict(list)
    for h in raw_hits:
        if h["instrument_group"] in STRATIFIED_GROUPS:
            all_by_instrument[h["instrument_group"]].append(h["velocity"])

    velocity_thresholds: dict[str, tuple[float, float]] = {}
    for instr, vels in all_by_instrument.items():
        low, high = np.percentile(vels, [33, 66])
        velocity_thresholds[instr] = (float(low), float(high))
        click.echo(f"  {instr} velocity tertiles: soft<{low:.1f}, medium<{high:.1f}, hard>={high:.1f}")

    # ------------------------------------------------------------------
    # 4a. Per-style buckets
    #     - stratified (kick/snare only): rock|{beat_type}|{instrument}|{tier}
    #     - unstratified (all instruments): rock|{beat_type}|{instrument}
    # ------------------------------------------------------------------
    # Group hits by (beat_type, instrument_group)
    style_buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    # Group stratified hits by (beat_type, instrument_group, tier)
    tier_buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for h in raw_hits:
        key = (h["beat_type"], h["instrument_group"])
        style_buckets[key].append(h)

        if h["instrument_group"] in STRATIFIED_GROUPS:
            thresholds = velocity_thresholds[h["instrument_group"]]
            tier = _velocity_tier(h["velocity"], thresholds)
            tier_buckets[(h["beat_type"], h["instrument_group"], tier)].append(h)

    # 4b. Global buckets: global|{instrument_group}
    global_buckets: dict[str, list[dict]] = defaultdict(list)
    for h in raw_hits:
        global_buckets[h["instrument_group"]].append(h)

    # ------------------------------------------------------------------
    # 5. Build profiles, enforcing MIN_SAMPLES threshold
    # ------------------------------------------------------------------
    profiles: dict[str, list[list[float]]] = {}
    written = 0
    skipped_buckets = 0

    # Stratified buckets (kick and snare only)
    for (beat_type, instrument_group, tier), hits in tier_buckets.items():
        if len(hits) < MIN_SAMPLES:
            skipped_buckets += 1
            continue
        key = f"rock|{beat_type}|{instrument_group}|{tier}"
        profiles[key] = _build_pairs(hits)
        written += 1

    # Unstratified per-style buckets (all instruments, also serves as tier-drop fallback)
    for (beat_type, instrument_group), hits in style_buckets.items():
        if len(hits) < MIN_SAMPLES:
            skipped_buckets += 1
            continue
        key = f"rock|{beat_type}|{instrument_group}"
        profiles[key] = _build_pairs(hits)
        written += 1

    # Global buckets (unstratified, fallback level 4 for kick/snare, level 3 for others)
    for instrument_group, hits in global_buckets.items():
        if len(hits) < MIN_SAMPLES:
            skipped_buckets += 1
            continue
        key = f"global|{instrument_group}"
        profiles[key] = _build_pairs(hits)
        written += 1

    # ------------------------------------------------------------------
    # 6. Write JSON (bucket data + _meta)
    # ------------------------------------------------------------------
    output: dict = {
        "_meta": {
            "velocity_thresholds": {
                instr: list(thresholds)
                for instr, thresholds in velocity_thresholds.items()
            },
            "kde_bw_method": KDE_BW_METHOD,
        }
    }
    output.update(profiles)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w") as f:
        json.dump(output, f)

    click.echo(
        f"Buckets written: {written}  skipped (< {MIN_SAMPLES} samples): {skipped_buckets}"
    )
    click.echo(f"Profile written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
