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
    detect_meter,
    grid_position_in_bar,
    is_four_four,
    quantise_to_grid,
)

EPSILON_TICKS = 1

# MIDI channel 10 (0-indexed 9) — the GM drum channel. Only notes on this channel are
# humanised by default; melodic parts often use drum-range note numbers and must not
# be treated as kit hits. Opt out with all_channels=True (--all-channels).
DRUM_CHANNEL = 9


@dataclasses.dataclass(frozen=True)
class BucketProfile:
    offsets: np.ndarray     # (N,) raw offset_ms values
    vel_deltas: np.ndarray  # (N,) raw vel_delta values
    kde: Any                # gaussian_kde fitted on (offset_ms, vel_delta) pairs, or None


@dataclasses.dataclass(frozen=True)
class LoadedProfile:
    buckets: dict[str, BucketProfile]
    velocity_thresholds: dict[str, tuple[float, float]]
    bucket_offset_means: dict[str, float] = dataclasses.field(default_factory=dict)


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

    bucket_offset_means: dict[str, float] = {
        k: float(v) for k, v in meta.get("bucket_offset_means", {}).items()
    }
    return LoadedProfile(
        buckets=buckets,
        velocity_thresholds=vel_thresholds,
        bucket_offset_means=bucket_offset_means,
    )


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
) -> tuple[BucketProfile | None, int | None, str | None]:
    """Return the best-matching BucketProfile, its fallback level (1-based), and the matched key.

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
            return profile.buckets[key], level, key
    return None, None, None


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
        # KDE resampling adds ~one bandwidth of Gaussian noise, so a draw can land
        # outside the 2nd–98th-percentile range the profile was clipped to at build
        # time — reintroducing exactly the tail outliers clipping removed. Clamp back
        # to the retained data range (already in memory) to restore that intent.
        offset = float(np.clip(sample[0, 0], bucket.offsets.min(), bucket.offsets.max()))
        vel_delta = float(np.clip(sample[1, 0], bucket.vel_deltas.min(), bucket.vel_deltas.max()))
        return offset, vel_delta
    else:
        idx = np.random.randint(len(bucket.offsets))
        return float(bucket.offsets[idx]), float(bucket.vel_deltas[idx])


# Groove-drift tuning. Fixed internal constants — phi ("groove tightness") is the only
# user-facing knob; calibrating these from data is a deferred step.
RESIDUAL_SHARE = 0.15          # β: the residual's share of a solo hit's timing variance
COUPLED_RESIDUAL_FRAC = 0.15   # coupled-hit scatter as a fraction of its own centred sample …
COUPLED_RESIDUAL_MS = 1.0      # … capped so each coupled member stays within ±1 ms of the shared nudge


class GrooveDrift:
    """One drummer's internal clock for the whole kit.

    A shifted *solo* hit advances an AR(1) drift and adds a small independent residual, so the
    kit's timing wanders together (correlated) instead of scattering hit-to-hit while the
    per-hit spread is preserved: for a stationary bucket ``Var(step()) == Var(c)``. One instance
    is shared across all tracks — hits are fed to it in absolute-time order regardless of which
    track they live on. Constructed and used only when ``phi > 0`` — ``phi == 0`` is an exact
    bypass in ``humanise()`` (no drift, no coupling), so this class never has to reproduce the
    legacy path.
    """

    def __init__(self, phi: float, rng: Any = None) -> None:
        self.phi = phi
        self._innov = math.sqrt(1.0 - phi * phi)          # AR(1) scale → stationary Var(drift)==Var(c)
        self._w_drift = math.sqrt(1.0 - RESIDUAL_SHARE)   # variance split weights (drift vs residual)
        self._w_resid = math.sqrt(RESIDUAL_SHARE)
        # Residual noise source. A dedicated RNG (not the global np.random used by
        # _sample_bucket) keeps the offset/velocity sample stream independent of phi, so
        # phi=0 and phi>0 draw the *same* samples and differ only in timing processing.
        self._rng = rng if rng is not None else np.random
        self.drift = 0.0

    def step(self, c: float, sigma: float) -> float:
        """Advance the clock by one solo hit and return its timing fluctuation (ms).

        ``c`` is the hit's mean-centred offset sample; ``sigma`` scales the fresh independent
        residual to the bucket's own spread so the split keeps ``Var(output) == Var(c)`` for a
        stationary (single-bucket) input — on a real mixed-instrument stream it is approximate.
        """
        self.drift = self.phi * self.drift + self._innov * c
        return self._w_drift * self.drift + self._w_resid * float(self._rng.normal(0.0, sigma))


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
    push: bool = False,
    phi: float = 0.4,
    all_channels: bool = False,
) -> None:
    if timing_only and velocity_only:
        raise ValueError("timing_only and velocity_only are mutually exclusive")
    if not (0.0 <= phi < 1.0):
        raise ValueError("phi (groove tightness) must be in [0.0, 1.0)")
    np.random.seed(seed)
    # Groove-residual RNG, seeded independently of (but reproducibly from) `seed` so it never
    # perturbs the global sample stream — offset/velocity draws stay identical across phi.
    resid_seed = None if seed is None else (seed + 2_246_822_519) % (2 ** 32)
    resid_rng = np.random.RandomState(resid_seed)

    mid = mido.MidiFile(str(input_path))
    if mid.type == 2:
        raise ValueError("Type 2 MIDI files are not supported")
    meter = detect_meter(mid)   # raises ValueError for 6/8 mixed with other signatures
    grid = "8" if meter == "6/8" else "16"
    # Grid-position buckets assume a 4/4 bar, so positional lookups only run on 4/4
    # files. Straight non-4/4 meters (3/4, 5/4, non-6/8 mixed-meter files) are
    # accepted with grid_pos=None — the 6/8 precedent — and fall back to the
    # per-instrument ms deviation buckets, which transfer well across meters.
    # detect_meter() above has already rejected 6/8-mixed files.
    use_grid_pos = meter != "6/8" and is_four_four(mid)
    out_mid = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    tempo_map = build_tempo_map(mid)
    ppq = mid.ticks_per_beat
    n_tracks = len(mid.tracks)

    # ---- Pass 1 (per track): absolute ticks, will_shift, windowing bounds ----
    tracks_abs: list[list[tuple[int, mido.Message]]] = []
    tracks_will_shift: list[list[bool]] = []
    tracks_next_fixed: list[list[float]] = []
    tracks_paired_off: list[list[float]] = []

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
                and (all_channels or msg.channel == DRUM_CHANNEL)
                and msg.note in TD11_TO_GROUP
            )
            if shiftable:
                gp = (grid_position_in_bar(quantise_to_grid(abs_t, ppq, grid), ppq)
                      if use_grid_pos else None)
                shiftable = (
                    _lookup(
                        profiles, genre, beat_type, TD11_TO_GROUP[msg.note],
                        msg.velocity, grid_pos=gp,
                    )[0] is not None  # only first element needed here
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

        tracks_abs.append(abs_messages)
        tracks_will_shift.append(will_shift)
        tracks_next_fixed.append(next_fixed)
        tracks_paired_off.append(paired_note_off_abs)

    # ---- Pass 2 (global): one absolute-time-ordered stream across all tracks ----
    # The groove clock and chord coupling are KIT-WIDE state. Multi-track drum MIDI
    # (kick/snare/hat exported on separate tracks — the normal DAW layout) must share
    # ONE drifting clock, and hits notated on the same tick must couple across tracks;
    # per-track clocks flammed cross-track accents by up to ~73 ms at phi=0.5.
    # Windowing and delta-time reconstruction stay per-track. For a single-track file
    # the merged order equals the original order, so output is byte-identical.
    merged = [
        (abs_t, ti, i)
        for ti, abs_msgs in enumerate(tracks_abs)
        for i, (abs_t, _msg) in enumerate(abs_msgs)
    ]
    merged.sort(key=lambda e: e[0])  # stable — ties keep track order, then message order

    out_abs: list[list[tuple[int, mido.Message]]] = [[] for _ in range(n_tracks)]
    prev_note_on_abs = [-EPSILON_TICKS] * n_tracks
    prev_emitted_abs = [0] * n_tracks
    # Original (pre-shift) tick of the most recent emitted note_on, per track. Used to
    # detect same-track chords: two note_ons notated at the same tick are simultaneous
    # hits and must be allowed to stay on the same tick, rather than being
    # force-separated by EPSILON_TICKS (which flams every kit-wide accent).
    prev_note_on_orig_abs = [-EPSILON_TICKS] * n_tracks
    # Groove-drift state: ONE internal clock for the whole kit. None at phi==0 (exact bypass).
    groove = GrooveDrift(phi, resid_rng) if phi != 0.0 else None
    chord_tick: int | None = None   # original tick of the current chord's anchor solo hit (any track)
    chord_anchor_abs = 0            # that anchor's ACTUAL emitted tick; coupled members land near it

    for abs_t, ti, idx in merged:
        msg = tracks_abs[ti][idx][1]
        if tracks_will_shift[ti][idx]:
            group = TD11_TO_GROUP[msg.note]
            grid_tick = quantise_to_grid(abs_t, ppq, grid)
            grid_pos = (grid_position_in_bar(grid_tick, ppq)
                        if use_grid_pos else None)
            bucket, level, key_used = _lookup(profiles, genre, beat_type, group, msg.velocity, grid_pos=grid_pos)
            offset_ms_raw, vel_delta_raw = _sample_bucket(bucket)

            if velocity_only:
                new_vel = max(1, min(127, round(msg.velocity + vel_delta_raw * intensity)))
                out_abs[ti].append((abs_t, msg.copy(velocity=new_vel)))
                prev_note_on_abs[ti] = abs_t
                prev_note_on_orig_abs[ti] = abs_t
                prev_emitted_abs[ti] = abs_t
                if verbose:
                    print(f"  note {msg.note} ({group}): level {level}")
                continue

            mu_debias = profiles.bucket_offset_means.get(key_used, 0.0)
            is_chord_anchor = False
            if phi == 0.0:
                # Exact pre-drift behaviour: drift and coupling both inert.
                offset_ms = offset_ms_raw - (0.0 if push else mu_debias)
                candidate = grid_tick + _ms_offset_to_ticks(
                    grid_tick, offset_ms * intensity, tempo_map, ppq
                )
            elif abs_t == chord_tick:
                # Coupled member (same original tick as the anchor solo hit, on ANY track):
                # land at the anchor's ACTUAL emitted tick plus a tiny ±COUPLED_RESIDUAL_MS
                # residual — NOT its pre-clamp desired offset — then let the window clamp to
                # this member's own legal range. Targeting the real landing keeps the chord
                # tight even when the anchor was window-clamped or a fixed same-tick event
                # sits between members. Does NOT advance the clock.
                c = offset_ms_raw - float(bucket.offsets.mean())
                residual_ms = max(-COUPLED_RESIDUAL_MS,
                                  min(COUPLED_RESIDUAL_MS, COUPLED_RESIDUAL_FRAC * c))
                candidate = chord_anchor_abs + _ms_offset_to_ticks(
                    chord_anchor_abs, residual_ms * intensity, tempo_map, ppq
                )
            else:
                # Solo hit / chord anchor: centre on the bucket's own mean (so a legacy/meanless
                # profile's lean is never amplified), advance the clock + independent residual,
                # then re-apply the systematic lean per the push contract.
                mu_center = float(bucket.offsets.mean())
                fluct = groove.step(offset_ms_raw - mu_center, float(bucket.offsets.std()))
                offset_ms = fluct + (mu_center - (0.0 if push else mu_debias))
                candidate = grid_tick + _ms_offset_to_ticks(
                    grid_tick, offset_ms * intensity, tempo_map, ppq
                )
                chord_tick = abs_t
                is_chord_anchor = True
            if timing_only:
                new_vel = msg.velocity
            else:
                new_vel = max(1, min(127, round(msg.velocity + vel_delta_raw * intensity)))

            if abs_t == prev_note_on_orig_abs[ti]:
                # Chord member (same original tick as the previous note_on on this track):
                # allow it to share prev_emitted_abs so the hit stays tight.
                lower = prev_emitted_abs[ti]
            else:
                lower = max(prev_emitted_abs[ti], prev_note_on_abs[ti] + EPSILON_TICKS)
            upper_exclusive = min(
                tracks_paired_off[ti][idx],
                tracks_next_fixed[ti][idx + 1] if idx + 1 < len(tracks_abs[ti]) else math.inf,
            )
            ceiling = upper_exclusive - 1  # math.inf - 1 == inf; safe

            if lower > ceiling:
                # No legal window: hold at prev_emitted_abs (guarantees non-negative
                # delta; same-tick with a fixed event is accepted as unavoidable).
                new_abs = prev_emitted_abs[ti]
            else:
                new_abs = int(max(lower, min(candidate, ceiling)))

            out_abs[ti].append((new_abs, msg.copy(velocity=new_vel)))
            prev_note_on_abs[ti] = new_abs
            prev_note_on_orig_abs[ti] = abs_t
            prev_emitted_abs[ti] = new_abs
            if is_chord_anchor:
                # Record the anchor's real landing so coupled members track it, not the
                # pre-clamp desired offset.
                chord_anchor_abs = new_abs
            if verbose:
                print(f"  note {msg.note} ({group}): level {level}")

        else:
            out_abs[ti].append((abs_t, msg))
            if msg.type == "note_on" and msg.velocity > 0:
                prev_note_on_abs[ti] = abs_t
                prev_note_on_orig_abs[ti] = abs_t
            prev_emitted_abs[ti] = abs_t

    # ---- Pass 3 (per track): convert abs ticks back to delta times ----
    for ti in range(n_tracks):
        new_track = mido.MidiTrack()
        prev = 0
        for abs_t, msg in out_abs[ti]:
            delta = abs_t - prev
            assert delta >= 0, "BUG: negative delta — likely invalid MIDI input"
            new_track.append(msg.copy(time=delta))
            prev = abs_t
        out_mid.tracks.append(new_track)

    out_mid.save(str(output_path))
