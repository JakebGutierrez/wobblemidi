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
    grid_position_in_bar,
    is_four_four,
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


# --- A2: velocity residualisation ------------------------------------------------
# vel_delta = velocity − mean velocity of the hit's own (take, grid_position,
# instrument) cell. Accent structure lives in the contour the USER programs; the old
# definition (delta vs bucket median) sampled that structure as noise. The residual
# keeps only within-performance imperfection (velocity sigma roughly halves).
#
# Guardrail 2 — sparse-cell shrinkage: an n=1 cell's mean equals the hit itself,
# making its residual exactly 0 (fake zero-noise crashes/fills). Every cell mean is
# shrunk toward the broader (take, instrument) mean with a pseudo-count:
#     shrunk = (n * cell_mean + SHRINKAGE_K * take_instr_mean) / (n + SHRINKAGE_K)
# SHRINKAGE_K = 5 is the threshold in effect: cells with n ≲ 5 are materially shrunk
# (an n=1 cell keeps 5/6 of its deviation from the take/instrument level as its
# residual); cells with n ≫ 5 are essentially untouched.
SHRINKAGE_K = 5


def residualise_velocities(raw_hits: list[dict]) -> None:
    """Annotate every hit with its vel_delta (in place) per the A2 definition above.

    Requires each hit to carry "take", "grid_pos", "instrument_group", "velocity".
    """
    cell_vels: dict[tuple, list[float]] = defaultdict(list)
    take_instr_vels: dict[tuple, list[float]] = defaultdict(list)
    for h in raw_hits:
        cell_vels[(h["take"], h["grid_pos"], h["instrument_group"])].append(h["velocity"])
        take_instr_vels[(h["take"], h["instrument_group"])].append(h["velocity"])

    shrunk_mean: dict[tuple, float] = {}
    for (take, pos, instr), vels in cell_vels.items():
        broad = float(np.mean(take_instr_vels[(take, instr)]))
        n = len(vels)
        shrunk_mean[(take, pos, instr)] = (
            n * float(np.mean(vels)) + SHRINKAGE_K * broad
        ) / (n + SHRINKAGE_K)

    for h in raw_hits:
        h["vel_delta"] = h["velocity"] - shrunk_mean[
            (h["take"], h["grid_pos"], h["instrument_group"])
        ]


def _build_pairs(hits: list[dict]) -> list[list[float]]:
    """Convert raw hit dicts to [[offset_ms, vel_delta], ...] pairs (A4 storage contract).

    vel_delta is the hit's velocity residual against its own shrunk
    (take, grid_position, instrument) mean, precomputed by residualise_velocities().
    """
    return [[h["offset_ms"], h["vel_delta"]] for h in hits]


def _clip_hits(hits: list[dict]) -> list[dict] | None:
    """Clip offset outliers and enforce MIN_SAMPLES.

    Returns the retained hit list (2nd–98th percentile of offset_ms), or None
    if fewer than MIN_SAMPLES hits remain after clipping.
    """
    offsets = np.array([h["offset_ms"] for h in hits])
    lo, hi = np.percentile(offsets, [2, 98])
    retained = [h for h in hits if lo <= h["offset_ms"] <= hi]
    if len(retained) < MIN_SAMPLES:
        return None
    return retained


def _build_pairs_with_clip(hits: list[dict]) -> list[list[float]] | None:
    """Clip offset outliers, enforce MIN_SAMPLES, then build [[offset_ms, vel_delta], ...].

    Drops hits with offset_ms outside the 2nd–98th percentile to remove
    accidental timing errors from GMD recordings before KDE fitting.
    Returns None if fewer than MIN_SAMPLES hits remain after clipping so the
    caller can skip the bucket and let the fallback chain handle it.
    """
    retained = _clip_hits(hits)
    if retained is None:
        return None
    return _build_pairs(retained)


def _build_profiles(
    grid_tier_buckets: dict,
    grid_style_buckets: dict,
    tier_buckets: dict,
    style_buckets: dict,
    global_buckets: dict,
) -> tuple[dict[str, list[list[float]]], dict[str, dict[str, float]], int, int]:
    """Build the profiles dict from pre-grouped bucket dicts.

    Returns (profiles, stats, written_count, skipped_count) where stats holds the
    per-bucket _meta maps: "bucket_offset_means", "bucket_vel_delta_means" (the bias
    removed by the guardrail-1 de-bias), and "vel_sigma_within".
    Every bucket family uses _clip_hits so all stats are computed from the same
    retained set used for KDE fitting.
    """
    profiles: dict[str, list[list[float]]] = {}
    stats: dict[str, dict[str, float]] = {
        "bucket_offset_means": {},
        "bucket_vel_delta_means": {},
        "vel_sigma_within": {},
    }
    written = 0
    skipped = 0

    def _write(key: str, hits: list[dict]) -> None:
        nonlocal written, skipped
        retained = _clip_hits(hits)
        if retained is None:
            skipped += 1
            return
        pairs = _build_pairs(retained)
        # Guardrail 1 (A2): residualising per (take, pos, instrument) does NOT give a
        # zero mean at the emitted-bucket level (tier buckets systematically collect
        # the low/high residuals of their cells, re-introducing accent bias). De-bias
        # every emitted bucket to ~0 and keep the removed mean in _meta for diagnostics.
        vd_mean = float(np.mean([p[1] for p in pairs]))
        pairs = [[off, vd - vd_mean] for off, vd in pairs]
        debiased = np.array([p[1] for p in pairs])
        assert abs(float(debiased.mean())) < 1e-9, (
            f"bucket {key}: vel_delta mean {float(debiased.mean()):.3e} not ~0 after de-bias"
        )
        profiles[key] = pairs
        stats["bucket_offset_means"][key] = float(np.mean([h["offset_ms"] for h in retained]))
        stats["bucket_vel_delta_means"][key] = vd_mean
        stats["vel_sigma_within"][key] = float(debiased.std())
        written += 1

    for (beat_type, instrument_group, tier, gp), hits in grid_tier_buckets.items():
        _write(f"rock|{beat_type}|{instrument_group}|{tier}|{gp}", hits)

    for (beat_type, instrument_group, gp), hits in grid_style_buckets.items():
        _write(f"rock|{beat_type}|{instrument_group}|{gp}", hits)

    for (beat_type, instrument_group, tier), hits in tier_buckets.items():
        _write(f"rock|{beat_type}|{instrument_group}|{tier}", hits)

    for (beat_type, instrument_group), hits in style_buckets.items():
        _write(f"rock|{beat_type}|{instrument_group}", hits)

    for instrument_group, hits in global_buckets.items():
        _write(f"global|{instrument_group}", hits)

    return profiles, stats, written, skipped


def collect_hits(gmd_dir: Path, files: pd.DataFrame) -> tuple[list[dict], int]:
    """Collect raw hits from the GMD takes listed in *files* (rows of info.csv).

    Returns (raw_hits, skipped_files). Missing, unreadable, and non-4/4 files
    are silently skipped with a counter.
    """
    raw_hits: list[dict] = []
    skipped_files = 0

    for row in files.itertuples():
        midi_path = gmd_dir / row.midi_filename
        if not midi_path.exists():
            skipped_files += 1
            continue

        try:
            midi = mido.MidiFile(str(midi_path))
        except Exception:
            skipped_files += 1
            continue

        if not is_four_four(midi):
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
                grid_pos = grid_position_in_bar(grid_tick, ppq)

                # Signed offset: positive = late, negative = early
                # Use ticks_to_ms_with_map for both legs so tempo changes
                # between the note and its grid point are handled correctly.
                if abs_tick >= grid_tick:
                    offset_ms = ticks_to_ms_with_map(grid_tick, abs_tick, tempo_map, ppq)
                else:
                    offset_ms = -ticks_to_ms_with_map(abs_tick, grid_tick, tempo_map, ppq)

                raw_hits.append(
                    {
                        "take": row.id,   # A1: source-take tag (unlocks A2 + split builds)
                        "beat_type": row.beat_type,
                        "instrument_group": instrument_group,
                        "offset_ms": offset_ms,
                        "velocity": float(msg.velocity),
                        "grid_pos": grid_pos,
                    }
                )

    return raw_hits, skipped_files


def build_profile_output(raw_hits: list[dict]) -> tuple[dict, int, int]:
    """Compute thresholds, group hits into buckets, clip, and assemble the profile dict.

    Returns (output, written, skipped_buckets) where *output* is the JSON-ready
    profile dict including the ``_meta`` block.
    """
    # A2: annotate every hit with its velocity residual before any bucketing.
    residualise_velocities(raw_hits)

    # ------------------------------------------------------------------
    # Compute velocity tertile thresholds for kick and snare
    # (from post-filter raw_hits so boundaries match the retained data)
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
    # Per-style buckets
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

    # Grid-position-aware buckets (all instruments)
    grid_style_buckets: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    grid_tier_buckets: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)

    for h in raw_hits:
        gp = h["grid_pos"]
        grid_style_buckets[(h["beat_type"], h["instrument_group"], gp)].append(h)
        if h["instrument_group"] in STRATIFIED_GROUPS:
            thresholds = velocity_thresholds[h["instrument_group"]]
            tier = _velocity_tier(h["velocity"], thresholds)
            grid_tier_buckets[(h["beat_type"], h["instrument_group"], tier, gp)].append(h)

    # Global buckets: global|{instrument_group}
    global_buckets: dict[str, list[dict]] = defaultdict(list)
    for h in raw_hits:
        global_buckets[h["instrument_group"]].append(h)

    # ------------------------------------------------------------------
    # Build profiles, clipping offset outliers and enforcing MIN_SAMPLES
    # ------------------------------------------------------------------
    profiles, stats, written, skipped_buckets = _build_profiles(
        grid_tier_buckets, grid_style_buckets, tier_buckets, style_buckets, global_buckets
    )

    # ------------------------------------------------------------------
    # Assemble JSON-ready dict (bucket data + _meta)
    # A4: bucket values stay [[offset_ms, vel_delta], ...]; only _meta grows.
    # ------------------------------------------------------------------
    output: dict = {
        "_meta": {
            "schema_version": 2,
            "vel_delta_definition": (
                "velocity residual vs shrunk (take, grid_position, instrument) mean; "
                "each emitted bucket de-biased to mean 0"
            ),
            "velocity_thresholds": {
                instr: list(thresholds)
                for instr, thresholds in velocity_thresholds.items()
            },
            "kde_bw_method": KDE_BW_METHOD,
            "bucket_offset_means": stats["bucket_offset_means"],
            "bucket_vel_delta_means": stats["bucket_vel_delta_means"],
            "vel_sigma_within": stats["vel_sigma_within"],
        }
    }
    output.update(profiles)

    return output, written, skipped_buckets


@click.command()
@click.argument("gmd_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--split", type=click.Choice(["all", "train"]), default="all", show_default=True,
              help="Restrict source takes to a GMD split (train = the validation "
                   "harness's gate split).")
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path),
              default=OUTPUT_FILE, show_default=True, help="Where to write the profile JSON.")
def main(gmd_dir: Path, split: str, output_path: Path) -> None:
    """Ingest GMD rock files and write timing/velocity profiles to pocketmidi/profiles/rock.json."""
    info = pd.read_csv(gmd_dir / "info.csv")
    rock = info[info["style"].str.startswith("rock")]
    if split == "train":
        rock = rock[rock["split"] == "train"]
    click.echo(f"Rock files: {len(rock)}" + (" [train split only]" if split == "train" else ""))

    raw_hits, skipped_files = collect_hits(gmd_dir, rock)
    click.echo(f"Total raw hits collected: {len(raw_hits)}  (files skipped: {skipped_files})")

    output, written, skipped_buckets = build_profile_output(raw_hits)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f)

    click.echo(
        f"Buckets written: {written}  skipped (< {MIN_SAMPLES} samples): {skipped_buckets}"
    )
    click.echo(f"Profile written to: {output_path}")


if __name__ == "__main__":
    main()
