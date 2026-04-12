"""Core humanisation engine: applies real-drummer timing/velocity distributions to drum MIDI."""

from __future__ import annotations

import bisect
import dataclasses
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import mido
import numpy as np
from scipy.stats import gaussian_kde

from pocketmidi.midi_utils import (
    TD11_TO_GROUP,
    build_tempo_map,
    grid_position_in_bar,
    is_four_four,
    quantise_to_grid,
)

EPSILON_TICKS = 1


@dataclasses.dataclass(frozen=True)
class BucketProfile:
    offsets: np.ndarray     # (N,) raw offset_ms values
    vel_deltas: np.ndarray  # (N,) raw vel_delta values
    kde: Any                # gaussian_kde fitted on (offset_ms, vel_delta) pairs, or None


@dataclasses.dataclass(frozen=True)
class LoadedProfile:
    buckets: dict[str, BucketProfile]
    velocity_thresholds: dict[str, tuple[float, float]]


def load_profile(path: str | Path) -> LoadedProfile:
    with open(path) as f:
        raw = json.load(f)

    meta = raw.pop("_meta", {})
    vel_thresholds: dict[str, tuple[float, float]] = {
        k: (float(v[0]), float(v[1]))
        for k, v in meta.get("velocity_thresholds", {}).items()
    }
    bw_method = meta.get("kde_bw_method", "scott")
    # Pre-validate bw_method so any bad value from _meta raises immediately rather
    # than being silently swallowed by the per-bucket degenerate-data guard below.
    # Valid values: the two scipy-accepted strings, or a numeric scalar bandwidth.
    if isinstance(bw_method, str):
        if bw_method not in {"scott", "silverman"}:
            raise ValueError(
                f"Invalid kde_bw_method {bw_method!r} in profile _meta; "
                "valid strings are 'scott' and 'silverman' (or use a numeric scalar)."
            )
    elif not isinstance(bw_method, (int, float)):
        raise ValueError(
            f"Invalid kde_bw_method {bw_method!r} in profile _meta; "
            "must be 'scott', 'silverman', or a numeric scalar."
        )

    buckets: dict[str, BucketProfile] = {}
    for key, pairs in raw.items():
        if not pairs:
            continue
        arr = np.array(pairs, dtype=float)   # shape (N, 2)
        offsets = arr[:, 0]
        vel_deltas = arr[:, 1]
        try:
            kde = gaussian_kde(arr.T, bw_method=bw_method)  # arr.T shape (2, N)
        except (ValueError, np.linalg.LinAlgError):
            # Degenerate bucket (too few samples or singular covariance). bw_method
            # was already validated above, so any ValueError here is a data issue.
            # Keep the bucket alive with kde=None; uniform-index fallback handles it.
            kde = None
        buckets[key] = BucketProfile(offsets=offsets, vel_deltas=vel_deltas, kde=kde)

    return LoadedProfile(buckets=buckets, velocity_thresholds=vel_thresholds)


def _velocity_tier(velocity: int, thresholds: tuple[float, float]) -> str:
    low, high = thresholds
    if velocity < low:
        return "soft"
    elif velocity < high:
        return "medium"
    else:
        return "hard"


def _lookup(
    profile: LoadedProfile,
    genre: str,
    beat_type: str,
    instrument_group: str,
    velocity: int,
    grid_pos: int | None = None,
) -> tuple[BucketProfile | None, int | None]:
    """Return the best-matching BucketProfile and its fallback level (1-based).

    When grid_pos is provided, stratified instruments try tier+grid_pos then
    unstratified+grid_pos before dropping to non-grid keys (offset=2).
    Unstratified instruments try instrument+grid_pos before non-grid keys (offset=1).
    When grid_pos is None, offset=0 and level numbering is identical to today.
    """
    thresholds = profile.velocity_thresholds.get(instrument_group)
    tier = _velocity_tier(velocity, thresholds) if thresholds else None

    if tier:
        # Up to 6-level chain for stratified instruments (kick, snare).
        # With grid_pos: try tier+grid_pos, then unstratified+grid_pos (keeps position
        # signal alive past a tier miss), then tier-only, then non-grid fallbacks.
        candidates: list[tuple[int, str]] = []
        offset = 0
        if grid_pos is not None:
            candidates.append((1, f"{genre}|{beat_type}|{instrument_group}|{tier}|{grid_pos}"))
            candidates.append((2, f"{genre}|{beat_type}|{instrument_group}|{grid_pos}"))
            offset = 2
        candidates += [
            (1 + offset, f"{genre}|{beat_type}|{instrument_group}|{tier}"),
            (2 + offset, f"{genre}|{beat_type}|{instrument_group}"),
            (3 + offset, f"{genre}|beat|{instrument_group}"),
            (4 + offset, f"global|{instrument_group}"),
        ]
    else:
        # Up to 4-level chain for hi-hat, cymbals, etc.
        candidates = []
        offset = 0
        if grid_pos is not None:
            candidates.append((1, f"{genre}|{beat_type}|{instrument_group}|{grid_pos}"))
            offset = 1
        candidates += [
            (1 + offset, f"{genre}|{beat_type}|{instrument_group}"),
            (2 + offset, f"{genre}|beat|{instrument_group}"),
            (3 + offset, f"global|{instrument_group}"),
        ]

    for level, key in candidates:
        if key in profile.buckets:
            return profile.buckets[key], level
    return None, None


def _ms_offset_to_ticks(
    grid_tick: int,
    offset_ms: float,
    tempo_map: list[tuple[int, int]],
    ppq: int,
) -> int:
    """Convert signed offset_ms from grid_tick into a signed tick delta."""
    if offset_ms == 0.0:
        return 0
    sign = 1 if offset_ms >= 0 else -1
    remaining = abs(offset_ms)
    current = grid_tick
    ticks_list = [t for t, _ in tempo_map]

    while remaining > 1e-9:
        idx = bisect.bisect_right(ticks_list, current) - 1

        if sign > 0:
            tempo_us = tempo_map[max(idx, 0)][1]
            ms_per_tick = tempo_us / ppq / 1_000.0
            if idx + 1 < len(tempo_map):
                ticks_to_next = tempo_map[idx + 1][0] - current
                ms_to_next = ticks_to_next * ms_per_tick
                if remaining <= ms_to_next:
                    current += round(remaining / ms_per_tick)
                    remaining = 0.0
                else:
                    remaining -= ms_to_next
                    current = tempo_map[idx + 1][0]
            else:
                current += round(remaining / ms_per_tick)
                remaining = 0.0
        else:
            # Backward walk: if current is exactly on a tempo boundary, step into
            # the preceding segment to avoid ticks_to_prev == 0 → infinite loop.
            while idx > 0 and current == tempo_map[idx][0]:
                idx -= 1
            tempo_us = tempo_map[max(idx, 0)][1]
            ms_per_tick = tempo_us / ppq / 1_000.0
            seg_start = tempo_map[max(idx, 0)][0]
            ticks_to_prev = current - seg_start
            ms_to_prev = ticks_to_prev * ms_per_tick
            if remaining <= ms_to_prev:
                current -= round(remaining / ms_per_tick)
                remaining = 0.0
            else:
                remaining -= ms_to_prev
                current = seg_start
                if current <= 0:
                    remaining = 0.0  # clamp at tick 0

    return current - grid_tick


def _sample_bucket(bucket: BucketProfile) -> tuple[float, float]:
    """Draw one (offset_ms, vel_delta) sample from a bucket.

    Primary path: joint 2D KDE sample.
    Fallback (kde is None — degenerate bucket): uniform draw from raw pairs.
    Both paths advance numpy RNG state by one draw so seed behaviour is consistent.
    """
    if bucket.kde is not None:
        sample = bucket.kde.resample(1)   # shape (2, 1); uses numpy RNG state
        return float(sample[0, 0]), float(sample[1, 0])
    else:
        idx = np.random.randint(len(bucket.offsets))
        return float(bucket.offsets[idx]), float(bucket.vel_deltas[idx])


def humanise(
    input_path: str | Path,
    output_path: str | Path,
    profiles: LoadedProfile,
    genre: str = "rock",
    beat_type: str = "beat",
    intensity: float = 1.0,
    seed: int | None = None,
    verbose: bool = False,
    timing_only: bool = False,
    velocity_only: bool = False,
) -> None:
    if timing_only and velocity_only:
        raise ValueError("timing_only and velocity_only are mutually exclusive")
    np.random.seed(seed)

    mid = mido.MidiFile(str(input_path))
    if mid.type == 2:
        raise ValueError("Type 2 MIDI files are not supported")
    # Only enforce 4/4 when the profile contains grid-position-aware buckets.
    # A grid-pos key ends with a numeric segment (e.g. "rock|beat|kick|hard|3").
    # Profiles without such keys fall back to the pre-grid-pos chain and work on
    # any time signature, so rejecting them here would be a regression.
    profile_has_grid_pos = any(
        key.split("|")[-1].isdigit() for key in profiles.buckets
    )
    if profile_has_grid_pos and not is_four_four(mid):
        raise ValueError(
            "Only 4/4 time is supported (the loaded profile contains grid-position "
            "buckets that assume a 4/4 bar length). "
            "Found a non-4/4 time_signature message in the MIDI file."
        )
    out_mid = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    tempo_map = build_tempo_map(mid)
    ppq = mid.ticks_per_beat
    out_tracks = []

    for track in mid.tracks:
        # Pass 1a — collect abs ticks (per-track, reset to 0)
        abs_messages: list[tuple[int, mido.Message]] = []
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            abs_messages.append((abs_tick, msg.copy(time=0)))

        N = len(abs_messages)

        # Pass 1b — precompute will_shift
        will_shift = []
        for abs_t, msg in abs_messages:
            shiftable = (
                msg.type == "note_on"
                and msg.velocity > 0
                and hasattr(msg, "note")
                and msg.note in TD11_TO_GROUP
            )
            if shiftable:
                gp = grid_position_in_bar(quantise_to_grid(abs_t, ppq), ppq)
                shiftable = (
                    _lookup(
                        profiles, genre, beat_type, TD11_TO_GROUP[msg.note],
                        msg.velocity, grid_pos=gp,
                    )[0] is not None
                )
            will_shift.append(shiftable)

        # next_fixed[i]: abs_tick of first event at j >= i where will_shift[j] is False
        next_fixed = [math.inf] * N
        last_fixed = math.inf
        for i in range(N - 1, -1, -1):
            if not will_shift[i]:
                last_fixed = abs_messages[i][0]
            next_fixed[i] = last_fixed

        # paired_note_off_abs[i]: FIFO-paired note_off abs_tick for each note_on
        open_note_ons: dict = defaultdict(deque)
        paired_note_off_abs = [math.inf] * N

        for i, (abs_t, msg) in enumerate(abs_messages):
            if msg.type == "note_on" and msg.velocity > 0:
                open_note_ons[(msg.note, msg.channel)].append(i)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if open_note_ons[key]:
                    note_on_idx = open_note_ons[key].popleft()
                    paired_note_off_abs[note_on_idx] = abs_t

        # Pass 2
        out_abs: list[tuple[int, mido.Message]] = []
        prev_note_on_abs = -EPSILON_TICKS
        prev_emitted_abs = 0

        for idx, (abs_t, msg) in enumerate(abs_messages):
            if will_shift[idx]:
                group = TD11_TO_GROUP[msg.note]
                grid_tick = quantise_to_grid(abs_t, ppq)
                grid_pos = grid_position_in_bar(grid_tick, ppq)
                bucket, level = _lookup(profiles, genre, beat_type, group, msg.velocity, grid_pos=grid_pos)
                offset_ms_raw, vel_delta_raw = _sample_bucket(bucket)

                if velocity_only:
                    new_vel = max(1, min(127, round(msg.velocity + vel_delta_raw * intensity)))
                    out_abs.append((abs_t, msg.copy(velocity=new_vel)))
                    prev_note_on_abs = abs_t
                    prev_emitted_abs = abs_t
                    if verbose:
                        print(f"  note {msg.note} ({group}): level {level}")
                    continue

                candidate = grid_tick + _ms_offset_to_ticks(
                    grid_tick, offset_ms_raw * intensity, tempo_map, ppq
                )
                if timing_only:
                    new_vel = msg.velocity
                else:
                    new_vel = max(1, min(127, round(msg.velocity + vel_delta_raw * intensity)))

                lower = max(prev_emitted_abs, prev_note_on_abs + EPSILON_TICKS)
                upper_exclusive = min(
                    paired_note_off_abs[idx],
                    next_fixed[idx + 1] if idx + 1 < N else math.inf,
                )
                ceiling = upper_exclusive - 1  # math.inf - 1 == inf; safe

                if lower > ceiling:
                    # No legal window: hold at prev_emitted_abs (guarantees non-negative
                    # delta; same-tick with a fixed event is accepted as unavoidable).
                    new_abs = prev_emitted_abs
                else:
                    new_abs = int(max(lower, min(candidate, ceiling)))

                out_abs.append((new_abs, msg.copy(velocity=new_vel)))
                prev_note_on_abs = new_abs
                prev_emitted_abs = new_abs
                if verbose:
                    print(f"  note {msg.note} ({group}): level {level}")

            else:
                out_abs.append((abs_t, msg))
                if msg.type == "note_on" and msg.velocity > 0:
                    prev_note_on_abs = abs_t
                prev_emitted_abs = abs_t

        # Convert abs ticks back to delta times
        new_track = mido.MidiTrack()
        prev = 0
        for abs_t, msg in out_abs:
            delta = abs_t - prev
            assert delta >= 0, "BUG: negative delta — likely invalid MIDI input"
            new_track.append(msg.copy(time=delta))
            prev = abs_t
        out_tracks.append(new_track)

    for t in out_tracks:
        out_mid.tracks.append(t)
    out_mid.save(str(output_path))
