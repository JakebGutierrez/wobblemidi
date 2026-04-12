"""Core humanisation engine: applies real-drummer timing/velocity distributions to drum MIDI."""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import mido
import numpy as np

from pocketmidi.midi_utils import TD11_TO_GROUP, build_tempo_map, quantise_to_grid

EPSILON_TICKS = 1

ProfileArrays = tuple[np.ndarray, np.ndarray]  # (offsets_ms, vel_deltas)


def load_profile(path: str | Path) -> dict[str, ProfileArrays]:
    with open(path) as f:
        raw = json.load(f)
    profiles: dict[str, ProfileArrays] = {}
    for key, pairs in raw.items():
        if not pairs:
            continue
        # Assumes well-formed pairs [[offset_ms, vel_delta], ...].
        # Add shape/type validation here when --profile (custom user paths) is implemented.
        arr = np.array(pairs)            # shape (N, 2)
        profiles[key] = (arr[:, 0], arr[:, 1])
    return profiles


def _lookup(
    profiles: dict[str, ProfileArrays],
    genre: str,
    beat_type: str,
    instrument_group: str,
) -> tuple[ProfileArrays | None, int | None]:
    for level, key in enumerate([
        f"{genre}|{beat_type}|{instrument_group}",
        f"{genre}|beat|{instrument_group}",
        f"global|{instrument_group}",
    ], start=1):
        if key in profiles:
            return profiles[key], level
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


def humanise(
    input_path: str | Path,
    output_path: str | Path,
    profiles: dict[str, ProfileArrays],
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
                and _lookup(profiles, genre, beat_type, TD11_TO_GROUP[msg.note])[0] is not None
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
                arrays, level = _lookup(profiles, genre, beat_type, group)
                offsets, vel_deltas = arrays
                i = np.random.randint(len(offsets))
                j = np.random.randint(len(vel_deltas))

                if velocity_only:
                    vel_delta = vel_deltas[j] * intensity
                    new_vel = max(1, min(127, round(msg.velocity + vel_delta)))
                    out_abs.append((abs_t, msg.copy(velocity=new_vel)))
                    prev_note_on_abs = abs_t
                    prev_emitted_abs = abs_t
                    if verbose:
                        print(f"  note {msg.note} ({group}): level {level}")
                    continue

                grid_tick = quantise_to_grid(abs_t, ppq)
                candidate = grid_tick + _ms_offset_to_ticks(
                    grid_tick, offsets[i] * intensity, tempo_map, ppq
                )
                if timing_only:
                    new_vel = msg.velocity
                else:
                    vel_delta = vel_deltas[j] * intensity
                    new_vel = max(1, min(127, round(msg.velocity + vel_delta)))

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
