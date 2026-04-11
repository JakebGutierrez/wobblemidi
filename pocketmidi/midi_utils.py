"""Low-level MIDI utilities: instrument mapping, tick/ms conversion, grid quantisation."""

from __future__ import annotations

import bisect
from typing import Sequence

# ---------------------------------------------------------------------------
# Roland TD-11 note → instrument group mapping
# Source: Magenta Groove MIDI Dataset drum mapping table
# ---------------------------------------------------------------------------
TD11_TO_GROUP: dict[int, str] = {
    36: "kick",
    38: "snare",
    40: "snare",
    37: "snare",
    48: "tom_high",
    50: "tom_high",
    45: "tom_mid",
    47: "tom_mid",
    43: "tom_low",
    58: "tom_low",
    42: "hihat_closed",
    44: "hihat_closed",
    22: "hihat_closed",   # HH Closed (Edge)
    46: "hihat_open",
    26: "hihat_open",     # HH Open (Edge)
    49: "crash",
    55: "crash",
    57: "crash",
    52: "crash",
    51: "ride",
    59: "ride",
    53: "ride",
}

# Default tempo: 120 BPM expressed as microseconds per beat
DEFAULT_TEMPO_US = 500_000


# ---------------------------------------------------------------------------
# Tempo map helpers
# ---------------------------------------------------------------------------

def build_tempo_map(midi_file) -> list[tuple[int, int]]:
    """Build a sorted list of (absolute_tick, tempo_us) pairs from a mido MidiFile.

    Scans all tracks for set_tempo MetaMessages and returns them in tick order.
    If no tempo event is found, defaults to 120 BPM (500 000 µs/beat).
    """
    events: list[tuple[int, int]] = []
    for track in midi_file.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                events.append((abs_tick, msg.tempo))
    events.sort(key=lambda x: x[0])
    if not events or events[0][0] != 0:
        events.insert(0, (0, DEFAULT_TEMPO_US))
    return events


def get_tempo_at_tick(tempo_map: Sequence[tuple[int, int]], tick: int) -> int:
    """Return the active tempo (µs/beat) at *tick*.

    Uses binary search over the sorted tempo_map.
    """
    ticks = [t for t, _ in tempo_map]
    idx = bisect.bisect_right(ticks, tick) - 1
    return tempo_map[max(idx, 0)][1]


# ---------------------------------------------------------------------------
# Tick ↔ ms conversion
# ---------------------------------------------------------------------------

def ticks_to_ms(ticks: int, tempo_us: int, ppq: int) -> float:
    """Convert a tick duration to milliseconds.

    Args:
        ticks:     Duration in ticks.
        tempo_us:  Tempo in microseconds per beat.
        ppq:       Ticks per beat (pulses per quarter note) from the MIDI header.

    Returns:
        Duration in milliseconds.
    """
    if ppq <= 0:
        raise ValueError(f"ppq must be positive, got {ppq}")
    return ticks * (tempo_us / ppq) / 1_000.0


def ticks_to_ms_with_map(
    start_tick: int,
    end_tick: int,
    tempo_map: Sequence[tuple[int, int]],
    ppq: int,
) -> float:
    """Convert a tick range [start_tick, end_tick) to milliseconds, respecting tempo changes.

    Splits the range at any tempo-change boundaries within it.
    """
    if ppq <= 0:
        raise ValueError(f"ppq must be positive, got {ppq}")
    ticks_list = [t for t, _ in tempo_map]
    ms = 0.0
    current = start_tick
    while current < end_tick:
        idx = bisect.bisect_right(ticks_list, current) - 1
        tempo_us = tempo_map[max(idx, 0)][1]
        # Find next tempo change
        next_change = end_tick
        if idx + 1 < len(tempo_map):
            next_change = min(tempo_map[idx + 1][0], end_tick)
        ms += ticks_to_ms(next_change - current, tempo_us, ppq)
        current = next_change
    return ms


# ---------------------------------------------------------------------------
# Grid quantisation
# ---------------------------------------------------------------------------

def quantise_to_grid(time_ticks: int, ppq: int) -> int:
    """Snap *time_ticks* to the nearest 16th-note grid position.

    Args:
        time_ticks:  Absolute tick position of a note.
        ppq:         Ticks per beat from the MIDI header.

    Returns:
        Nearest 16th-note grid position in ticks.
    """
    if ppq <= 0:
        raise ValueError(f"ppq must be positive, got {ppq}")
    sixteenth = ppq // 4
    remainder = time_ticks % sixteenth
    if remainder < sixteenth // 2:
        return time_ticks - remainder
    else:
        return time_ticks - remainder + sixteenth


def offset_ticks_to_ms(
    note_ticks: int,
    grid_ticks: int,
    tempo_us: int,
    ppq: int,
) -> float:
    """Return the timing deviation of a note from its grid position in milliseconds.

    Positive = late (note after grid), negative = early (note before grid).
    """
    delta_ticks = note_ticks - grid_ticks
    return ticks_to_ms(abs(delta_ticks), tempo_us, ppq) * (1 if delta_ticks >= 0 else -1)
