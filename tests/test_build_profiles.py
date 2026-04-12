import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_profiles import MIN_SAMPLES, _build_pairs_with_clip, _build_profiles


# ── unit tests for _build_pairs_with_clip ────────────────────────────────────

def test_outlier_offsets_absent_from_pairs():
    """Extreme offset_ms values are not present in the returned pairs."""
    normal = [{"offset_ms": float(i % 20 - 10), "velocity": 80.0} for i in range(98)]
    outliers = [
        {"offset_ms": -9999.0, "velocity": 10.0},
        {"offset_ms":  9999.0, "velocity": 10.0},
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
    hits = [{"offset_ms": 0.0, "velocity": 80.0} for _ in range(MIN_SAMPLES - 2)]
    hits += [
        {"offset_ms": -9999.0, "velocity": 80.0},
        {"offset_ms":  9999.0, "velocity": 80.0},
    ]
    assert _build_pairs_with_clip(hits) is None


def test_vel_delta_from_retained_set():
    """vel_delta is computed from the retained set only.

    All normal hits have velocity=80; after clipping the outlier hits
    (velocity=10) are gone, so every vel_delta must be exactly 0.
    If clipping were skipped the outlier pairs (delta = 10-80 = -70)
    would appear and the assertion would fail.
    """
    normal = [{"offset_ms": float(i % 20 - 10), "velocity": 80.0} for i in range(98)]
    outliers = [
        {"offset_ms": -9999.0, "velocity": 10.0},
        {"offset_ms":  9999.0, "velocity": 10.0},
    ]
    pairs = _build_pairs_with_clip(normal + outliers)
    assert pairs is not None
    assert all(d == 0.0 for p in pairs for d in [p[1]])


# ── integration test for _build_profiles ─────────────────────────────────────

def test_all_bucket_families_clip_outliers():
    """Every bucket family routes through _build_pairs_with_clip.

    If any loop called _build_pairs(hits) directly the extreme offset_ms
    values (-9999, 9999) would appear in the returned profile, failing the
    per-key assertion below.
    """
    n = MIN_SAMPLES + 5
    normal = [{"offset_ms": float(i % 10), "velocity": 80.0} for i in range(n)]
    extreme = [
        {"offset_ms": -9999.0, "velocity": 80.0},
        {"offset_ms":  9999.0, "velocity": 80.0},
    ]
    hits = normal + extreme

    profiles, written, _ = _build_profiles(
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
