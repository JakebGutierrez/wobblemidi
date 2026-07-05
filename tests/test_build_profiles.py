import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_profiles import (
    MIN_SAMPLES,
    SHRINKAGE_K,
    _build_pairs_with_clip,
    _build_profiles,
    _clip_hits,
    build_profile_output,
    residualise_velocities,
)


def _hits(n, offset_fn=lambda i: float(i % 20 - 10), vel_delta=0.0):
    """Hits with precomputed vel_delta (the A2 residual) as _build_pairs expects."""
    return [{"offset_ms": offset_fn(i), "velocity": 80.0, "vel_delta": vel_delta}
            for i in range(n)]


# ── unit tests for _build_pairs_with_clip ────────────────────────────────────

def test_outlier_offsets_absent_from_pairs():
    """Extreme offset_ms values are not present in the returned pairs."""
    normal = _hits(98)
    outliers = [
        {"offset_ms": -9999.0, "velocity": 10.0, "vel_delta": -70.0},
        {"offset_ms":  9999.0, "velocity": 10.0, "vel_delta": -70.0},
    ]
    pairs = _build_pairs_with_clip(normal + outliers)
    assert pairs is not None
    offsets = [p[0] for p in pairs]
    assert -9999.0 not in offsets
    assert  9999.0 not in offsets


def test_returns_none_when_clip_drops_below_min_samples():
    """Returns None rather than writing an under-supported bucket."""
    # MIN_SAMPLES - 2 normal hits plus 2 extreme outliers: after clipping,
    # only MIN_SAMPLES - 2 remain, which is below the threshold.
    hits = _hits(MIN_SAMPLES - 2, offset_fn=lambda i: 0.0)
    hits += [
        {"offset_ms": -9999.0, "velocity": 80.0, "vel_delta": 0.0},
        {"offset_ms":  9999.0, "velocity": 80.0, "vel_delta": 0.0},
    ]
    assert _build_pairs_with_clip(hits) is None


def test_vel_delta_from_retained_set():
    """vel_delta pairs come from the retained set only.

    All normal hits have vel_delta=0; the outlier hits carry vel_delta=-70 and
    extreme offsets. After clipping they are gone, so every pair's vel_delta
    must be exactly 0. If clipping were skipped the -70 deltas would appear.
    """
    normal = _hits(98)
    outliers = [
        {"offset_ms": -9999.0, "velocity": 10.0, "vel_delta": -70.0},
        {"offset_ms":  9999.0, "velocity": 10.0, "vel_delta": -70.0},
    ]
    pairs = _build_pairs_with_clip(normal + outliers)
    assert pairs is not None
    assert all(p[1] == 0.0 for p in pairs)


# ── integration test for _build_profiles ─────────────────────────────────────

def test_all_bucket_families_clip_outliers():
    """Every bucket family routes through the clip step.

    If any loop called _build_pairs(hits) directly the extreme offset_ms
    values (-9999, 9999) would appear in the returned profile, failing the
    per-key assertion below.
    """
    n = MIN_SAMPLES + 5
    normal = _hits(n, offset_fn=lambda i: float(i % 10))
    extreme = [
        {"offset_ms": -9999.0, "velocity": 80.0, "vel_delta": 0.0},
        {"offset_ms":  9999.0, "velocity": 80.0, "vel_delta": 0.0},
    ]
    hits = normal + extreme

    profiles, _, written, _ = _build_profiles(
        grid_tier_buckets  = {("beat", "kick", "hard", 0): hits},
        grid_style_buckets = {("beat", "kick", 0): hits},
        tier_buckets       = {("beat", "kick", "hard"): hits},
        style_buckets      = {("beat", "kick"): hits},
        global_buckets     = {"kick": hits},
    )

    assert written == 5, "all five bucket families should have written a profile"
    for key, pairs in profiles.items():
        offsets = [p[0] for p in pairs]
        assert -9999.0 not in offsets, f"extreme offset survived in {key}"
        assert  9999.0 not in offsets, f"extreme offset survived in {key}"


# ── unit tests for _clip_hits and per-bucket _meta stats ─────────────────────

def test_clip_hits_returns_retained_set():
    """_clip_hits removes outliers and returns the retained hit list."""
    normal = _hits(98)
    outliers = [{"offset_ms": -9999.0, "velocity": 10.0, "vel_delta": 0.0},
                {"offset_ms": 9999.0, "velocity": 10.0, "vel_delta": 0.0}]
    retained = _clip_hits(normal + outliers)
    assert retained is not None
    assert all(h["offset_ms"] != -9999.0 for h in retained)
    assert all(h["offset_ms"] != 9999.0 for h in retained)


def test_clip_hits_returns_none_below_min_samples():
    """_clip_hits returns None when retained set falls below MIN_SAMPLES."""
    hits = _hits(MIN_SAMPLES - 2, offset_fn=lambda i: 0.0)
    hits += [{"offset_ms": -9999.0, "velocity": 80.0, "vel_delta": 0.0},
             {"offset_ms": 9999.0, "velocity": 80.0, "vel_delta": 0.0}]
    assert _clip_hits(hits) is None


def test_bucket_offset_means_written():
    """_build_profiles writes bucket_offset_means for all five bucket families."""
    n = MIN_SAMPLES + 5
    hits = _hits(n, offset_fn=lambda i: 10.0)
    _, stats, written, _ = _build_profiles(
        grid_tier_buckets  = {("beat", "kick", "hard", 0): hits},
        grid_style_buckets = {("beat", "kick", 0): hits},
        tier_buckets       = {("beat", "kick", "hard"): hits},
        style_buckets      = {("beat", "kick"): hits},
        global_buckets     = {"kick": hits},
    )
    assert written == 5
    means = stats["bucket_offset_means"]
    assert len(means) == 5
    for mean in means.values():
        assert abs(mean - 10.0) < 1e-6


# ── A2 guardrail 1: every emitted bucket is de-biased to ~0 mean ─────────────

def test_emitted_buckets_debiased_to_zero_mean():
    """Buckets whose input residuals carry a bias must come out with mean ~0,
    and the removed bias must be recorded in stats (the _meta diagnostic)."""
    n = MIN_SAMPLES + 10
    # residuals alternating around +5 — a biased bucket, as a soft/hard tier
    # bucket would be after (take, pos, instrument) residualisation
    hits = [{"offset_ms": float(i % 7 - 3), "velocity": 80.0,
             "vel_delta": 5.0 + (1.0 if i % 2 else -1.0)} for i in range(n)]
    profiles, stats, written, _ = _build_profiles(
        grid_tier_buckets={}, grid_style_buckets={},
        tier_buckets={("beat", "snare", "soft"): hits},
        style_buckets={}, global_buckets={},
    )
    assert written == 1
    key = "rock|beat|snare|soft"
    deltas = np.array([p[1] for p in profiles[key]])
    assert abs(deltas.mean()) < 1e-9
    np.testing.assert_allclose(stats["bucket_vel_delta_means"][key], 5.0, atol=0.2)
    assert stats["vel_sigma_within"][key] > 0


# ── A2 residualisation + guardrail 2 shrinkage ───────────────────────────────

def _raw_hit(take, pos, instr, vel):
    return {"take": take, "grid_pos": pos, "instrument_group": instr,
            "beat_type": "beat", "offset_ms": 0.0, "velocity": float(vel)}


def test_residual_is_deviation_from_cell_mean():
    """Dense cells: vel_delta ≈ velocity − (take, pos, instrument) cell mean.
    (hihat: not tier-conditioned, so this locks the base A2 path.)"""
    # one cell with many hits alternating 70/90 (mean 80); take-level mean also 80,
    # so shrinkage is a no-op and residuals must be exactly ±10
    hits = [_raw_hit("t1", 0, "hihat_closed", 70 if i % 2 else 90) for i in range(200)]
    residualise_velocities(hits)
    deltas = {h["vel_delta"] for h in hits}
    assert deltas == {10.0, -10.0}


# ── addendum Fix 1: snare tier-residualisation ───────────────────────────────

def test_snare_roles_get_separate_baselines():
    """Ghost/backbeat alternating at ONE position across bars: the tier-agnostic
    mean (65) would give residuals ±35 — the snare bug. With tier conditioning
    each role centres near itself, so residuals collapse. The identical hihat
    part keeps the blended behaviour (snare-only change)."""
    snare = [_raw_hit("t1", 4, "snare", 30 if i % 2 else 100) for i in range(64)]
    hat = [_raw_hit("t1", 4, "hihat_closed", 30 if i % 2 else 100) for i in range(64)]
    hits = snare + hat
    residualise_velocities(hits)
    snare_d = [abs(h["vel_delta"]) for h in hits if h["instrument_group"] == "snare"]
    hat_d = [abs(h["vel_delta"]) for h in hits if h["instrument_group"] == "hihat_closed"]
    assert max(snare_d) < 6, f"snare residuals still blended: max {max(snare_d):.1f}"
    assert min(hat_d) > 30, "hihat must keep the tier-agnostic baseline"


def test_snare_roles_are_relative_not_absolute():
    """Ghosts 60 / backbeats 100 both sit in the SAME absolute tier (51..111 =
    'medium'), so absolute tiers would blend them. The relative per-take role
    convention (same as runtime B4) must still separate them."""
    hits = [_raw_hit("t1", 4, "snare", 60 if i % 2 else 100) for i in range(64)]
    residualise_velocities(hits, velocity_thresholds={"snare": (51.0, 111.0)})
    assert max(abs(h["vel_delta"]) for h in hits) < 6


def test_snare_sparse_tier_cell_shrinks_tier_preserving():
    """A lone ghost at a new position must shrink toward the (take, snare, soft)
    tier mean (≈30), NOT toward the blended take mean (≈57) — the Codex
    tier-preserving fallback. Straight-to-blended would leave |residual| ≈ 23."""
    hits = [_raw_hit("t1", p, "snare", 30) for p in range(1, 9) for _ in range(3)]
    hits += [_raw_hit("t1", 12, "snare", 100) for _ in range(16)]
    lone = _raw_hit("t1", 0, "snare", 30)
    hits.append(lone)
    residualise_velocities(hits)
    assert abs(lone["vel_delta"]) < 10, (
        f"lone ghost fell back to a blended mean: delta {lone['vel_delta']:.1f}"
    )


def test_snare_single_level_part_unchanged_behaviour():
    """All snare hits at one velocity: no role evidence and no absolute
    thresholds → single shared role → identical to the tier-agnostic path."""
    hits = [_raw_hit("t1", p % 4, "snare", 90) for p in range(40)]
    residualise_velocities(hits)
    assert all(h["vel_delta"] == 0.0 for h in hits)


def test_sparse_cell_residual_not_fake_zero():
    """Guardrail 2: an n=1 cell (lone crash) must NOT produce a zero residual —
    its cell mean is shrunk toward the (take, instrument) mean."""
    hits = [_raw_hit("t1", p, "crash", 60) for p in range(1, 9) for _ in range(3)]
    hits.append(_raw_hit("t1", 0, "crash", 110))   # lone loud crash, its own cell
    residualise_velocities(hits)
    lone = hits[-1]
    # unshrunk: residual would be exactly 0. With SHRINKAGE_K pseudo-counts of the
    # take/instrument mean, most of the 110-vs-60ish deviation must survive.
    assert lone["vel_delta"] > 30.0
    # exact shrinkage arithmetic: an n=1 cell keeps K/(1+K) of its deviation
    expected_frac = SHRINKAGE_K / (1 + SHRINKAGE_K)
    take_mean = float(np.mean([h["velocity"] for h in hits]))
    assert abs(lone["vel_delta"] - expected_frac * (110.0 - take_mean)) < 1e-9


def test_residuals_do_not_leak_across_takes():
    """Cell means are per take: identical positions in different takes get their
    own means (a loud take and a quiet take each centre on themselves)."""
    loud = [_raw_hit("loud", 0, "kick", v) for v in (100, 110)] * 20
    quiet = [_raw_hit("quiet", 0, "kick", v) for v in (60, 70)] * 20
    hits = loud + quiet
    residualise_velocities(hits)
    assert abs(np.mean([h["vel_delta"] for h in hits if h["take"] == "loud"])) < 1e-9
    assert abs(np.mean([h["vel_delta"] for h in hits if h["take"] == "quiet"])) < 1e-9
    # residual magnitudes reflect within-take variation (±5), not the 40-unit take gap
    assert all(abs(h["vel_delta"]) < 10 for h in hits)


# ── shipped-profile schema regression ────────────────────────────────────────

def test_shipped_profile_is_schema_v2():
    """The bundled rock.json must be a v2-builder artifact — guards against a
    stale or old-schema profile ever shipping again (module-13 ship contract)."""
    shipped = Path(__file__).parent.parent / "wobblemidi" / "profiles" / "rock.json"
    with shipped.open() as f:
        raw = json.load(f)
    meta = raw["_meta"]
    assert meta["schema_version"] == 2
    assert meta["tier_residual_groups"] == ["snare"]
    bucket_keys = {k for k in raw if k != "_meta"}
    # the new per-bucket _meta maps cover every emitted bucket
    assert set(meta["bucket_vel_delta_means"]) == bucket_keys
    assert set(meta["vel_sigma_within"]) == bucket_keys
    assert set(meta["bucket_offset_means"]) == bucket_keys
    # guardrail 1 holds in the shipped artifact: buckets are de-biased to ~0
    for key in ("global|snare", "global|kick", "global|hihat_closed"):
        deltas = np.array(raw[key])[:, 1]
        assert abs(float(deltas.mean())) < 1e-9, f"{key} not de-biased"


# ── build_profile_output end-to-end: schema v2 _meta contract ────────────────

def test_build_profile_output_schema_v2():
    rng = np.random.RandomState(0)
    raw_hits = []
    for take in ("a", "b", "c"):
        for pos in range(16):
            for _ in range(4):
                raw_hits.append({
                    "take": take, "beat_type": "beat", "instrument_group": "kick",
                    "offset_ms": float(rng.normal(0, 10)),
                    "velocity": float(rng.randint(60, 100)), "grid_pos": pos,
                })
    output, written, _ = build_profile_output(raw_hits)

    meta = output["_meta"]
    assert meta["schema_version"] == 2
    assert meta["tier_residual_groups"] == ["snare"]
    assert "kick" in meta["velocity_thresholds"]
    assert set(meta["bucket_vel_delta_means"]) == set(meta["vel_sigma_within"])
    assert written > 0
    # A4: bucket values are still [[offset_ms, vel_delta], ...] pairs,
    # and every emitted bucket's vel_delta mean is ~0 (guardrail 1)
    for key, pairs in output.items():
        if key == "_meta":
            continue
        arr = np.array(pairs)
        assert arr.ndim == 2 and arr.shape[1] == 2
        assert abs(arr[:, 1].mean()) < 1e-9, f"bucket {key} not de-biased"
