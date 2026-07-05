"""Core humanisation engine: applies real-drummer timing/velocity distributions to drum MIDI."""

from __future__ import annotations

import bisect
import dataclasses
import json
import math
import sys
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
    ticks_to_ms_with_map,
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


# Relative velocity tiering (module 13 / spec B4). GMD-absolute tier thresholds
# collapse for programmed parts whose velocities all sit inside one GMD tertile —
# every hit routes to the same tier bucket regardless of the part's own dynamics.
# With enough evidence, tiers come from the user's own per-instrument distribution.
RELATIVE_TIER_MIN_HITS = 8      # need >= 8 hits of the instrument in the file ...
RELATIVE_TIER_MIN_SPREAD = 12   # ... and p90-p10 >= 12 velocity units to trust relative tiers
TWO_CLUSTER_MIN_FRAC = 0.15     # each side of a two-cluster split must hold >= 15% of hits
TWO_CLUSTER_DOMINANCE = 1.5     # the gap must be >= 1.5x either side's spread — it has to
                                # DWARF within-cluster spread (ghost/accent), so a 3-level
                                # evenly-spaced part still gets tertiles, not soft/hard


def _file_tier_thresholds(
    velocities: Any, absolute: tuple[float, float]
) -> tuple[float, float]:
    """Per-file (low, high) tier thresholds for one instrument (spec B4).

    - Insufficient evidence (few hits / narrow spread) → the profile's GMD-absolute
      thresholds, unchanged behaviour.
    - Exactly two distinct velocity values (a 2-level palette or a ghost/backbeat
      part) → soft/hard at their midpoint regardless of balance: the spread gate
      above already established a real gap, and an imbalanced duplicate part
      (e.g. 14 ghosts + 2 accents) must not fall through to tertiles that
      collapse onto the dominant value and route the ghosts to hard.
    - Two-cluster (ghost/accent) parts → both thresholds at the midpoint of the
      dominant velocity gap, so hits map to soft/hard only — no bogus medium.
      A gap qualifies when it is at least as wide as the spread of either side it
      separates and both sides hold a real share of the hits.
    - Otherwise → relative tertiles (33rd/66th percentile of the file's own
      velocities for that instrument), tie-aware: a dominant duplicated value can
      collapse both tertiles onto itself, so collapsed thresholds are re-anchored
      to midpoints BETWEEN distinct values, never on one.
    Thresholds compare values, so equal velocities always share a tier (ties preserved).
    """
    v = np.asarray(velocities, dtype=float)
    if len(v) < RELATIVE_TIER_MIN_HITS:
        return absolute
    p10, p90 = np.percentile(v, [10, 90])
    if p90 - p10 < RELATIVE_TIER_MIN_SPREAD:
        return absolute
    vals = np.unique(v)
    if len(vals) == 2:
        boundary = float(vals.mean())
        return (boundary, boundary)
    gaps = np.diff(vals)
    gi = int(np.argmax(gaps))
    gap = float(gaps[gi])
    frac_lo = float((v <= vals[gi]).mean())
    spread_lo = float(vals[gi] - vals[0])
    spread_hi = float(vals[-1] - vals[gi + 1])
    if (
        gap >= RELATIVE_TIER_MIN_SPREAD
        and gap >= TWO_CLUSTER_DOMINANCE * max(spread_lo, spread_hi)
        and min(frac_lo, 1.0 - frac_lo) >= TWO_CLUSTER_MIN_FRAC
    ):
        boundary = float(vals[gi]) + gap / 2.0
        return (boundary, boundary)
    low, high = np.percentile(v, [33, 66])
    if low == high:
        # Tie-aware re-anchor: both tertiles collapsed onto one dominant value t*
        # (>= a third of the mass on each side of it is t* itself). Boundaries on a
        # data value misroute it — v < low means t* and everything below it would go
        # NOT-soft. Place each boundary halfway to the adjacent distinct value; a
        # missing side (t* is the bottom/top level) collapses soft/hard accordingly.
        t_star = float(low)
        below = vals[vals < t_star]
        above = vals[vals > t_star]
        lo_bound = float((below[-1] + t_star) / 2.0) if len(below) else None
        hi_bound = float((t_star + above[0]) / 2.0) if len(above) else None
        # spread gate guarantees at least one neighbour exists
        if lo_bound is None:
            return (hi_bound, hi_bound)   # t* is the bottom level → soft
        if hi_bound is None:
            return (lo_bound, lo_bound)   # t* is the top level → hard
        return (lo_bound, hi_bound)       # t* is the middle level → medium
    return (float(low), float(high))


def _lookup(
    profile: LoadedProfile,
    genre: str,
    beat_type: str,
    instrument_group: str,
    velocity: int,
    grid_pos: int | None = None,
    thresholds: tuple[float, float] | None = None,
) -> tuple[BucketProfile | None, int | None, str | None]:
    """Return the best-matching BucketProfile, its fallback level (1-based), and the matched key.

    When grid_pos is provided, stratified instruments try tier+grid_pos then
    unstratified+grid_pos before dropping to non-grid keys (offset=2).
    Unstratified instruments try instrument+grid_pos before non-grid keys (offset=1).
    When grid_pos is None, offset=0 and level numbering is identical to today.

    ``thresholds`` overrides the profile's GMD-absolute tier thresholds — humanise()
    passes the per-file relative thresholds from _file_tier_thresholds (spec B4), so
    the tier selects both the velocity AND timing bucket by the user's own dynamics.
    None (the default, and any direct caller) keeps the absolute behaviour.
    """
    if thresholds is None:
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

# Time-windowed coupling (phi > 0 only). Shiftable hits whose REAL elapsed time
# from a cluster's first hit is <= this window (inclusive; measured through the
# tempo map — PPQ alone is not time) move as ONE rigid unit: close-spaced
# ornaments (snare flams, grace notes) must not be scattered or smeared by
# independent sampling. Conservative by design; tunable by ear, NOT a CLI knob.
COUPLE_WINDOW_MS = 12.0

# Velocity drift (module 13 / spec B2). GMD kick velocity has genuine hit-to-hit
# memory (train-split lag-1 r=+0.317 → PHI_VEL = r/(1-RESIDUAL_SHARE), see
# scripts/calibrate_phi.py); snare/hats/ride/crash are ~white (r 0.07–0.16), so
# i.i.d. sampling stays correct for them. Kick-only, and never shared across
# instruments — a same-tick crash/snare keeps its own independent draw.
# Fixed internal tuning, deliberately NOT a CLI knob.
PHI_VEL = 0.37
VEL_DRIFT_GROUPS = frozenset({"kick"})


class GrooveDrift:
    """AR(1) drift plus independent residual — the engine's slow-wander building block.

    Each ``step`` advances an AR(1) drift on the mean-centred sample and adds a small
    independent residual, so successive values wander together (correlated) instead of
    scattering hit-to-hit while the per-hit spread is preserved: for a stationary bucket
    ``Var(step()) == Var(c)``.

    Used for two kit-wide clocks in ``humanise()``:
    * the TIMING clock (``phi`` = --groove-tightness): advanced by every shifted *solo*
      hit across all tracks in absolute-time order. Constructed only when ``phi > 0`` —
      ``phi == 0`` is an exact timing bypass (no drift, no coupling), so this class
      never has to reproduce the legacy timing path.
    * the kick VELOCITY clock (``PHI_VEL``, spec B2): advanced by every kick hit whose
      velocity is humanised. Kick-only — no cross-instrument sharing.
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
    # Default 0.35: ear-tested sweet spot. 1.0 reproduces GMD's raw within-take spread
    # (~27 ms timing sigma) — real, but reads as sloppy in produced music. The scale
    # stays linear and un-capped; 1.0 remains fully available.
    intensity: float = 0.35,
    seed: int | None = None,
    verbose: bool = False,
    timing_only: bool = False,
    velocity_only: bool = False,
    push: bool = False,
    phi: float = 0.4,
    all_channels: bool = False,
    # Continuous lean (GUI round 3). None → derived from the `push` bool
    # (True→1.0, False→0.0), so all existing callers are unchanged. -1 MIRRORS
    # the stored per-bucket lean (inverts the source drummers' tendencies) —
    # it is NOT a synthetic "laid-back drag"; the shipped profile is not
    # uniformly early (ride leans late), so a=-1 flips each bucket's habit.
    push_amount: float | None = None,
    # Per-lane humanisation amount (output gain on that lane's deviation),
    # ABSOLUTE per group: eff(group) = intensity_by_group.get(group, intensity).
    # This is NOT independent per-lane timing feel — the kit still shares ONE
    # drift clock (a 0.0 lane still advances it), and same-tick chords are
    # governed by the tightest limb (min eff at that tick).
    intensity_by_group: dict[str, float] | None = None,
) -> None:
    if timing_only and velocity_only:
        raise ValueError("timing_only and velocity_only are mutually exclusive")
    if not (0.0 <= phi < 1.0):
        raise ValueError("phi (groove tightness) must be in [0.0, 1.0)")
    if push_amount is not None:
        if push:
            raise ValueError("push and push_amount are mutually exclusive")
        if not (-1.0 <= push_amount <= 1.0):
            raise ValueError("push_amount (lean) must be in [-1.0, 1.0]")
        lean = float(push_amount)
    else:
        lean = 1.0 if push else 0.0
    if intensity_by_group:
        valid_groups = set(TD11_TO_GROUP.values())
        unknown = set(intensity_by_group) - valid_groups
        if unknown:
            raise ValueError(
                f"unknown instrument group(s) in intensity_by_group: {sorted(unknown)}"
            )
        for g, v in intensity_by_group.items():
            # explicit finite check: nan slips through a bare `v < 0` (nan >= 0
            # is False but so is nan < 0), and inf would silently blow up ticks
            if not math.isfinite(v) or v < 0:
                raise ValueError(
                    f"intensity_by_group[{g!r}] must be a finite value >= 0, got {v}"
                )
    if push_amount is not None and push_amount != 1.0 and not profiles.bucket_offset_means:
        # Legacy (no-means) profile: there is no stored lean to remove/scale/mirror,
        # so any lean setting is a silent no-op. Say so once rather than nothing.
        print(
            "pocketmidi: note — this profile stores no per-bucket lean means "
            "(legacy schema); lean/push_amount has no effect.",
            file=sys.stderr,
        )
    np.random.seed(seed)
    # Groove-residual RNG, seeded independently of (but reproducibly from) `seed` so it never
    # perturbs the global sample stream — offset/velocity draws stay identical across phi.
    resid_seed = None if seed is None else (seed + 2_246_822_519) % (2 ** 32)
    resid_rng = np.random.RandomState(resid_seed)
    # Velocity-drift RNG: a third independent stream (spec B2). Keeping it separate from
    # both the global sample stream and resid_rng preserves the two existing contracts —
    # samples identical across phi, and timing identical across timing_only/default —
    # while the kick velocity clock draws its own residuals.
    vel_seed = None if seed is None else (seed + 1_779_033_703) % (2 ** 32)
    vel_drift = GrooveDrift(PHI_VEL, np.random.RandomState(vel_seed))

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

    # B4: per-file relative tier thresholds from the user's own velocities, for every
    # instrument the profile stratifies. Falls back to the profile's GMD-absolute
    # thresholds inside _file_tier_thresholds when evidence is insufficient.
    vels_by_group: dict[str, list[int]] = defaultdict(list)
    for track in mid.tracks:
        for msg in track:
            if (
                msg.type == "note_on"
                and msg.velocity > 0
                and hasattr(msg, "note")
                and (all_channels or msg.channel == DRUM_CHANNEL)
                and msg.note in TD11_TO_GROUP
            ):
                vels_by_group[TD11_TO_GROUP[msg.note]].append(msg.velocity)
    file_thresholds: dict[str, tuple[float, float]] = {
        g: _file_tier_thresholds(vs, profiles.velocity_thresholds[g])
        for g, vs in vels_by_group.items()
        if g in profiles.velocity_thresholds
    }

    # Per-lane output gain. When intensity_by_group is unset this returns the plain
    # `intensity` object, so every multiplication below is bit-identical to before.
    def _eff(group: str) -> float:
        if intensity_by_group:
            return intensity_by_group.get(group, intensity)
        return intensity

    # Per-original-tick MIN eff across all shiftable hits at that tick (kit-wide,
    # across tracks — the same keying chords use). The tightest limb governs a
    # chord's landing: a kick at eff 0.15 under a hat at 0.8 keeps the accent
    # anchored near the grid instead of being dragged out by the looser lane.
    # Filled during pass 1 below; only populated when intensity_by_group is set.
    tick_min_eff: dict[int, float] = {}

    def _new_velocity(group: str, bucket: BucketProfile, vel_delta_raw: float,
                      velocity: int) -> int:
        # B1: the sampled delta perturbs the user's own dynamics (shape unchanged).
        # B2: kick-only velocity drift. Centre on the bucket's own vel_delta mean so a
        # legacy (non-residual) profile's bias is applied statically and never amplified
        # — the same guard the timing clock uses. Other instruments: i.i.d. sample.
        if group in VEL_DRIFT_GROUPS:
            mu_v = float(bucket.vel_deltas.mean())
            fluct_v = vel_drift.step(vel_delta_raw - mu_v, float(bucket.vel_deltas.std()))
            vel_delta = fluct_v + mu_v
        else:
            vel_delta = vel_delta_raw
        return max(1, min(127, round(velocity + vel_delta * _eff(group))))

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
                        thresholds=file_thresholds.get(TD11_TO_GROUP[msg.note]),
                    )[0] is not None  # only first element needed here
                )
            if shiftable and intensity_by_group:
                e = _eff(TD11_TO_GROUP[msg.note])
                cur = tick_min_eff.get(abs_t)
                tick_min_eff[abs_t] = e if cur is None else min(cur, e)
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

    # ---- time-windowed coupling clusters (phi > 0 only; COUPLE_WINDOW_MS) ----
    # Purely additive generalisation of same-tick coupling: consecutive shiftable
    # hits within the window of the cluster's FIRST hit form one cluster. Only
    # clusters with a nonzero internal span take the new rigid-unit path below —
    # same-tick chords keep the existing anchor+residual behaviour byte-for-byte
    # and singletons stay solo hits. phi == 0 disables coupling entirely, as ever.
    cluster_of: dict[tuple[int, int], dict] = {}
    cluster_samples: dict[tuple[int, int], tuple[float, float]] = {}
    if groove is not None:
        def _close_cluster(run: list[tuple[int, int, int, int]]) -> None:
            if len(run) < 2 or run[-1][0] == run[0][0]:
                return                       # singleton or zero-gap: existing paths
            members = [(a, t, i) for a, t, i, _pos in run]
            # loudest member sources the timing sample — the main stroke leads,
            # the grace note follows. Ties: max() keeps the earliest in order.
            loudest = max(members,
                          key=lambda m: tracks_abs[m[1]][m[2]][1].velocity)
            min_eff = (min(_eff(TD11_TO_GROUP[tracks_abs[t][i][1].note])
                           for _a, t, i in members)
                       if intensity_by_group else intensity)
            cl = {
                "members": members, "loudest": loudest, "min_eff": min_eff,
                "mstart": run[0][3], "mend": run[-1][3],
                "planned": False, "delta": 0,
            }
            for _a, t, i in members:
                cluster_of[(t, i)] = cl

        _run: list[tuple[int, int, int, int]] = []
        for _pos, (_abs_e, _ti_e, _idx_e) in enumerate(merged):
            if not tracks_will_shift[_ti_e][_idx_e]:
                continue
            if _run and ticks_to_ms_with_map(
                    _run[0][0], _abs_e, tempo_map, ppq) <= COUPLE_WINDOW_MS:
                _run.append((_abs_e, _ti_e, _idx_e, _pos))
            else:
                _close_cluster(_run)
                _run = [(_abs_e, _ti_e, _idx_e, _pos)]
        _close_cluster(_run)

    def _plan_cluster(cl: dict, first_member: tuple[int, int],
                      first_sample: tuple[float, float]) -> None:
        """Fix a windowed cluster's single shared tick delta, once, at its first
        member's turn.

        Draws the remaining members' samples eagerly IN MERGED ORDER (members are
        consecutive shiftable hits, so the global RNG stream is unchanged), steps
        the kit clock ONCE on the loudest member's sample, then clamps the shared
        delta at CLUSTER scope: each member's legal interval against its own fixed
        context (note-offs, fixed events, prior emitted state) is intersected and
        the one delta clamped into it. Members are never clamped independently —
        that is what would collapse/distort a flam. Member-vs-member spacing needs
        no constraint: the shared delta preserves original gaps (>= EPSILON_TICKS).
        """
        samples: dict[tuple[int, int], tuple[float, float]] = {first_member: first_sample}
        for abs_m, ti_m, idx_m in cl["members"]:
            key_m = (ti_m, idx_m)
            if key_m == first_member:
                continue
            msg_m = tracks_abs[ti_m][idx_m][1]
            group_m = TD11_TO_GROUP[msg_m.note]
            grid_m = quantise_to_grid(abs_m, ppq, grid)
            gp_m = grid_position_in_bar(grid_m, ppq) if use_grid_pos else None
            bucket_m, _lvl_m, _key_used_m = _lookup(
                profiles, genre, beat_type, group_m, msg_m.velocity,
                grid_pos=gp_m, thresholds=file_thresholds.get(group_m),
            )
            pair = _sample_bucket(bucket_m)
            samples[key_m] = pair
            cluster_samples[key_m] = pair

        abs_l, ti_l, idx_l = cl["loudest"]
        msg_l = tracks_abs[ti_l][idx_l][1]
        group_l = TD11_TO_GROUP[msg_l.note]
        grid_l = quantise_to_grid(abs_l, ppq, grid)
        gp_l = grid_position_in_bar(grid_l, ppq) if use_grid_pos else None
        bucket_l, _lvl_l, key_l = _lookup(
            profiles, genre, beat_type, group_l, msg_l.velocity,
            grid_pos=gp_l, thresholds=file_thresholds.get(group_l),
        )
        offset_raw_l = samples[(ti_l, idx_l)][0]
        mu_center_l = float(bucket_l.offsets.mean())
        fluct = groove.step(offset_raw_l - mu_center_l, float(bucket_l.offsets.std()))
        mu_debias_l = profiles.bucket_offset_means.get(key_l, 0.0)
        offset_ms_l = fluct + (mu_center_l - (1.0 - lean) * mu_debias_l)
        cand_l = grid_l + _ms_offset_to_ticks(
            grid_l, offset_ms_l * cl["min_eff"], tempo_map, ppq
        )
        desired = cand_l - abs_l

        member_set = {(t, i) for _a, t, i in cl["members"]}
        sim_pe: dict[int, int] = {}
        sim_pn: dict[int, float] = {}
        sim_pno: dict[int, float] = {}
        d_lo, d_hi = -math.inf, math.inf     # musical interval (EPSILON, ceiling-1)
        h_lo, h_hi = -math.inf, math.inf     # HARD encodability interval (delta>=0 in pass 3)
        for pos in range(cl["mstart"], cl["mend"] + 1):
            abs_e, ti_e, idx_e = merged[pos]
            if (ti_e, idx_e) in member_set:
                pe = sim_pe.get(ti_e, prev_emitted_abs[ti_e])
                pn = sim_pn.get(ti_e, prev_note_on_abs[ti_e])
                pno = sim_pno.get(ti_e, prev_note_on_orig_abs[ti_e])
                lower = pe if abs_e == pno else max(pe, pn + EPSILON_TICKS)
                upper_ex = min(
                    tracks_paired_off[ti_e][idx_e],
                    tracks_next_fixed[ti_e][idx_e + 1]
                    if idx_e + 1 < len(tracks_abs[ti_e]) else math.inf,
                )
                d_lo = max(d_lo, lower - abs_e)
                d_hi = min(d_hi, (upper_ex - 1) - abs_e)
                h_lo = max(h_lo, pe - abs_e)
                h_hi = min(h_hi, upper_ex - abs_e)
                sim_pno[ti_e] = abs_e
            elif not tracks_will_shift[ti_e][idx_e]:
                # fixed event between members: replay exactly what the real loop
                # will do to this track's window state before the member's turn
                msg_e = tracks_abs[ti_e][idx_e][1]
                sim_pe[ti_e] = abs_e
                if msg_e.type == "note_on" and msg_e.velocity > 0:
                    sim_pn[ti_e] = abs_e
                    sim_pno[ti_e] = abs_e

        if d_lo <= d_hi:
            delta = int(max(d_lo, min(float(desired), d_hi)))
            rigid = True
        else:
            # Musically-empty intersection: the cluster HOLDS as a unit. Written
            # positions (delta 0) are the target, nudged forward uniformly only
            # as far as hard encodability demands (a prior hit may already have
            # emitted past a member's written tick — a negative MIDI delta is a
            # crash, so pure delta 0 is not always available). The EPSILON
            # separation and ceiling-1 preferences are sacrificed here: a held
            # flam is acceptable, a smeared one is not.
            delta = int(max(0.0, h_lo))
            rigid = delta <= h_hi
            if not rigid:
                # Truly unsatisfiable (zero-length notes + a late prior hit):
                # no shared delta is even encodable. Degrade to the legacy
                # per-member windowing below — crash-free, smear accepted,
                # pathological input only.
                delta = 0
        cl["delta"] = delta
        cl["rigid"] = rigid
        cl["planned"] = True

    for abs_t, ti, idx in merged:
        msg = tracks_abs[ti][idx][1]
        if tracks_will_shift[ti][idx]:
            group = TD11_TO_GROUP[msg.note]
            grid_tick = quantise_to_grid(abs_t, ppq, grid)
            grid_pos = (grid_position_in_bar(grid_tick, ppq)
                        if use_grid_pos else None)
            bucket, level, key_used = _lookup(
                profiles, genre, beat_type, group, msg.velocity, grid_pos=grid_pos,
                thresholds=file_thresholds.get(group),
            )
            stashed = cluster_samples.pop((ti, idx), None)
            if stashed is not None:
                # drawn eagerly (in stream order) when this hit's cluster was planned
                offset_ms_raw, vel_delta_raw = stashed
            else:
                offset_ms_raw, vel_delta_raw = _sample_bucket(bucket)

            if velocity_only:
                new_vel = _new_velocity(group, bucket, vel_delta_raw, msg.velocity)
                out_abs[ti].append((abs_t, msg.copy(velocity=new_vel)))
                prev_note_on_abs[ti] = abs_t
                prev_note_on_orig_abs[ti] = abs_t
                prev_emitted_abs[ti] = abs_t
                if verbose:
                    print(f"  note {msg.note} ({group}): level {level}")
                continue

            cl = cluster_of.get((ti, idx))
            if cl is not None:
                # windowed (gap > 0) cluster member: one rigid shared delta for
                # the whole cluster, fixed at the first member's turn. No
                # per-member residual and no per-member clamping. Velocity is
                # still per-hit (same samples, same vel-clock order as ever).
                if not cl["planned"]:
                    _plan_cluster(cl, (ti, idx), (offset_ms_raw, vel_delta_raw))
                if timing_only:
                    new_vel = msg.velocity
                else:
                    new_vel = _new_velocity(group, bucket, vel_delta_raw, msg.velocity)
                if cl["rigid"]:
                    # written position + the ONE shared delta, full stop. The
                    # cluster-scope clamp (or the hold fallback) already chose a
                    # delta every member can legally take — a per-member guard
                    # here is exactly what would smear the flam.
                    new_abs = abs_t + cl["delta"]
                else:
                    # pathological escape hatch (no shared delta is encodable):
                    # legacy per-member windowing — crash-free, smear accepted
                    candidate = abs_t + cl["delta"]
                    if abs_t == prev_note_on_orig_abs[ti]:
                        lower = prev_emitted_abs[ti]
                    else:
                        lower = max(prev_emitted_abs[ti],
                                    prev_note_on_abs[ti] + EPSILON_TICKS)
                    upper_exclusive = min(
                        tracks_paired_off[ti][idx],
                        tracks_next_fixed[ti][idx + 1]
                        if idx + 1 < len(tracks_abs[ti]) else math.inf,
                    )
                    ceiling = upper_exclusive - 1
                    if lower > ceiling:
                        new_abs = prev_emitted_abs[ti]
                    else:
                        new_abs = int(max(lower, min(candidate, ceiling)))
                out_abs[ti].append((new_abs, msg.copy(velocity=new_vel)))
                prev_note_on_abs[ti] = new_abs
                prev_note_on_orig_abs[ti] = abs_t
                prev_emitted_abs[ti] = new_abs
                chord_tick = None   # a windowed cluster supersedes same-tick coupling
                if verbose:
                    print(f"  note {msg.note} ({group}): level {level} [cluster]")
                continue

            mu_debias = profiles.bucket_offset_means.get(key_used, 0.0)
            eff_own = _eff(group)
            is_chord_anchor = False
            if phi == 0.0:
                # Exact pre-drift behaviour: drift and coupling both inert.
                # lean generalises the old push bool: (1-lean)*mu is bit-equal to
                # the old `0.0 if push else mu_debias` at lean ∈ {1.0, 0.0}.
                offset_ms = offset_ms_raw - (1.0 - lean) * mu_debias
                candidate = grid_tick + _ms_offset_to_ticks(
                    grid_tick, offset_ms * eff_own, tempo_map, ppq
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
                # Member residual scales by its OWN eff (a 0.0 lane sits exactly on
                # the anchor's landing); the anchor itself was already governed by
                # the tick's min eff, so the chord stays tight at any mix of gains.
                candidate = chord_anchor_abs + _ms_offset_to_ticks(
                    chord_anchor_abs, residual_ms * eff_own, tempo_map, ppq
                )
            else:
                # Solo hit / chord anchor: centre on the bucket's own mean (so a legacy/meanless
                # profile's lean is never amplified), advance the clock + independent residual,
                # then re-apply the systematic lean per the push contract.
                mu_center = float(bucket.offsets.mean())
                # The clock steps on the UNSCALED centred sample: the kit shares one
                # drift trajectory regardless of per-lane gains (a 0.0 lane still
                # drives it). Per-lane eff scales only this hit's OUTPUT deviation,
                # and an anchor is governed by the tick's min eff (tightest limb).
                fluct = groove.step(offset_ms_raw - mu_center, float(bucket.offsets.std()))
                offset_ms = fluct + (mu_center - (1.0 - lean) * mu_debias)
                anchor_eff = (tick_min_eff.get(abs_t, eff_own)
                              if intensity_by_group else intensity)
                candidate = grid_tick + _ms_offset_to_ticks(
                    grid_tick, offset_ms * anchor_eff, tempo_map, ppq
                )
                chord_tick = abs_t
                is_chord_anchor = True
            if timing_only:
                new_vel = msg.velocity   # velocity untouched; the vel clock does not advance
            else:
                new_vel = _new_velocity(group, bucket, vel_delta_raw, msg.velocity)

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
