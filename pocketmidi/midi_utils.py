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

def quantise_to_grid(time_ticks: int, ppq: int, grid: str = "16") -> int:
    """Snap *time_ticks* to the nearest grid position.

    Args:
        time_ticks:  Absolute tick position of a note.
        ppq:         Ticks per beat from the MIDI header.
        grid:        Grid resolution: ``"16"`` for 16th notes (default),
                     ``"8"`` for 8th notes (compound meters such as 6/8).

    Returns:
        Nearest grid position in ticks.
    """
    if ppq <= 0:
        raise ValueError(f"ppq must be positive, got {ppq}")
    subdivision = ppq // 4 if grid == "16" else ppq // 2
    remainder = time_ticks % subdivision
    if remainder < subdivision // 2:
        return time_ticks - remainder
    else:
        return time_ticks - remainder + subdivision


def grid_position_in_bar(grid_tick: int, ppq: int) -> int:
    """16th-note index within a 4/4 bar (0–15).

    Position 0 = beat 1 downbeat, 4 = beat 2, 8 = beat 3, 12 = beat 4.
    Assumes 4/4 time (hardcoded for v1; GMD rock dataset is 4/4 throughout).
    """
    sixteenth = ppq // 4          # matches quantise_to_grid() — integer division
    ticks_per_bar = 16 * sixteenth  # bar length on the quantised grid, not ppq * 4
    return (grid_tick % ticks_per_bar) // sixteenth


def is_four_four(midi_file) -> bool:
    """Return True if all time_signature messages in the file are 4/4.

    MIDI default when no time_signature is present is 4/4, so files with no
    time_signature events return True.
    """
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == "time_signature":
                if msg.numerator != 4 or msg.denominator != 4:
                    return False
    return True


def detect_meter(midi_file) -> str:
    """Return ``"6/8"`` if the file is uniformly 6/8 throughout; ``"non-6/8"`` otherwise.

    Collects all ``time_signature`` meta-messages with their absolute tick positions.
    If the first explicit 6/8 event is not at tick 0, there is an implicit 4/4 region
    before it — this counts as mixing 6/8 with 4/4 and raises ``ValueError``.

    Returns:
        ``"6/8"``      — all ``time_signature`` events are 6/8 and the first is at tick 0.
        ``"non-6/8"``  — no 6/8 events present (includes no events, uniform 4/4, uniform
                         3/4, and any non-6/8 mixed-meter files).

    Raises:
        ValueError — the file mixes 6/8 with any other time signature, including an
        implicit 4/4 prefix before the first 6/8 event.  All other mixed-meter files
        (e.g. 4/4 + 3/4) return ``"non-6/8"`` without raising, since the 16th-note grid
        is valid for any quarter-note-based meter.
    """
    events: list[tuple[int, int, int]] = []   # (abs_tick, numerator, denominator)
    for track in midi_file.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "time_signature":
                events.append((abs_tick, msg.numerator, msg.denominator))

    if not events:
        return "non-6/8"    # implicit 4/4 throughout

    events.sort(key=lambda x: x[0])
    has_six_eight = any(num == 6 and den == 8 for _, num, den in events)

    if not has_six_eight:
        return "non-6/8"

    # File contains at least one 6/8 event — any mixing is unsupported.
    if events[0][0] != 0:
        raise ValueError(
            "Mixed time signatures (implicit 4/4, 6/8) are not supported. "
            "The file must use a single time signature throughout."
        )
    non_six_eight = sorted({(num, den) for _, num, den in events
                             if not (num == 6 and den == 8)})
    if non_six_eight:
        sigs = ", ".join(f"{n}/{d}" for n, d in sorted({(6, 8)} | set(non_six_eight)))
        raise ValueError(
            f"Mixed time signatures ({sigs}) are not supported. "
            "The file must use a single time signature throughout."
        )
    return "6/8"


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
