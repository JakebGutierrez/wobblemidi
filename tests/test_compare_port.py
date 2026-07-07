"""Tests for scripts/compare_port.py — the Tier 2 port-vs-reference runner.

Two layers:
  * mechanics — fixture integrity, cell derivation, extraction/alignment,
    structural checks, scoring behaviour on identical/perturbed pools;
  * the calibrated-verdict locks (self-null must PASS, a gross mutant must
    FAIL) — added with the locked thresholds in calibration/, and skipped
    while no thresholds file exists so the runner commit stands alone.
"""

import shutil
import sys
from pathlib import Path

import mido
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import compare_port as cp
from verify_golden import load_bundled_profile, load_manifest

from wobblemidi.midi_utils import TD11_TO_GROUP


@pytest.fixture(scope="module")
def profile():
    prof, sha = load_bundled_profile()
    return prof


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def cells(manifest):
    return cp.load_cells(manifest)


# ── full-kit coverage fixture ────────────────────────────────────────────────

def test_full_kit_fixture_matches_builder():
    """The checked-in fixture is exactly what the builder produces — the runner
    refuses to trust a modified fixture, so this must hold in CI too."""
    assert cp.FULL_KIT_FIXTURE.exists(), \
        "run: python scripts/compare_port.py fixture"
    assert cp.FULL_KIT_FIXTURE.read_bytes() == cp.full_kit_bytes()


def test_full_kit_fixture_covers_all_td11_notes():
    mid = mido.MidiFile(str(cp.FULL_KIT_FIXTURE))
    counts: dict[int, int] = {}
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                assert msg.channel == 9
                counts[msg.note] = counts.get(msg.note, 0) + 1
    assert set(counts) == set(TD11_TO_GROUP), "every TD-11 note must appear"
    assert min(counts.values()) >= 4
    group_counts: dict[str, int] = {}
    for note, n in counts.items():
        g = TD11_TO_GROUP[note]
        group_counts[g] = group_counts.get(g, 0) + n
    assert set(group_counts) == set(TD11_TO_GROUP.values())
    assert min(group_counts.values()) >= 8


# ── cell derivation ──────────────────────────────────────────────────────────

def test_cells_cover_manifest_minus_skips_plus_full_kit(cells, manifest):
    ids = {c.id for c in cells}
    manifest_ids = {e["id"] for e in manifest["vectors"]}
    assert cp.FULL_KIT_CELL_ID in ids
    assert ids - {cp.FULL_KIT_CELL_ID} == manifest_ids - set(cp.SKIPPED_VECTORS)
    for c in cells:
        assert "seed" not in c.params
        assert c.input_path.exists()
    by_id = {c.id: c for c in cells}
    assert by_id["f9_empty_default"].kind == "identity"
    assert by_id["f1_intensity00"].kind == "identity"
    assert by_id["f1_default"].kind == "dist"
    # the full-kit cell runs at the same pinned params as f1_default
    assert by_id[cp.FULL_KIT_CELL_ID].params == by_id["f1_default"].params


def test_prepare_cell_full_kit(cells):
    cd = cp.prepare_cell(next(c for c in cells if c.id == cp.FULL_KIT_CELL_ID))
    assert len(cd.notes) == 240
    # every group pools >= MIN_POOLED_HITS at K_RUNS → all 9 groups gated
    assert set(cd.gated_groups) == set(TD11_TO_GROUP.values())
    assert cd.timing_scored and cd.velocity_scored
    # roles are input-derived and per-group
    assert set(np.unique(cd.roles)) <= {"soft", "medium", "hard", "all"}


def test_prepare_cell_mode_flags(cells):
    by_id = {c.id: c for c in cells}
    assert not cp.prepare_cell(by_id["f2_timing_only"]).velocity_scored
    assert cp.prepare_cell(by_id["f2_timing_only"]).timing_scored
    assert not cp.prepare_cell(by_id["f2_velocity_only"]).timing_scored
    assert cp.prepare_cell(by_id["f2_velocity_only"]).velocity_scored


def test_prepare_cell_window_pairs_flam_fixture(cells):
    """f4's 8-tick flam graces (≈8.3 ms at 120 BPM) are inside the 12 ms window."""
    by_id = {c.id: c for c in cells}
    cd = cp.prepare_cell(by_id["f4_default"])
    assert len(cd.window_pairs) > 0
    for i, j in cd.window_pairs:
        gap = cd.in_t[j] - cd.in_t[i]
        assert 0.0 < gap <= cp.WINDOW_MS
    # plain on-grid fixtures have no sub-12ms consecutive gaps
    assert cp.prepare_cell(by_id["f1_default"]).window_pairs == []


# ── extraction, alignment, structural checks ─────────────────────────────────

def test_reference_run_roundtrip(cells, profile, tmp_path):
    """A reference-engine output aligns 1:1 and produces zero violations."""
    by_id = {c.id: c for c in cells}
    cd = cp.prepare_cell(by_id["f7_default"])
    out = tmp_path / "o.mid"
    from wobblemidi.humanise import humanise
    humanise(cd.cell.input_path, out, profile, seed=1, **cd.cell.params)
    out_abs, out_vel, out_off, viol = cp.read_output(out, cd)
    assert viol == []
    assert len(out_abs) == len(cd.notes)
    assert len(out_off) == len(cd.off_abs)


def test_read_output_detects_misalignment(cells, tmp_path):
    """Feeding one cell's output shape into another cell's expectations fails
    structurally instead of producing garbage metrics."""
    by_id = {c.id: c for c in cells}
    cd_f1 = cp.prepare_cell(by_id["f1_default"])
    _, _, _, viol = cp.read_output(by_id["f3_default"].input_path, cd_f1)
    assert any("misalignment" in v or "track count" in v for v in viol)


def test_read_output_flags_melodic_channel_tampering(cells, tmp_path):
    """Moving a non-drum-channel note is a structural violation (f7 bass)."""
    by_id = {c.id: c for c in cells}
    cd = cp.prepare_cell(by_id["f7_default"])
    mid = mido.MidiFile(str(cd.cell.input_path))
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0 and msg.channel != 9:
                msg.velocity = max(1, msg.velocity - 10)   # tamper with the bass
    p = tmp_path / "tampered.mid"
    mid.save(str(p))
    _, _, _, viol = cp.read_output(p, cd)
    assert any("non-drum-channel" in v for v in viol)


def test_mode_contract_violations_detected(cells, profile, tmp_path):
    """A default-mode output violates both timing_only and velocity_only cells'
    structural contracts (positions AND velocities moved)."""
    by_id = {c.id: c for c in cells}
    from wobblemidi.humanise import humanise
    out = tmp_path / "o.mid"
    cd_t = cp.prepare_cell(by_id["f2_timing_only"])
    humanise(cd_t.cell.input_path, out, profile, seed=1,
             **by_id["f2_default"].params)     # default mode: velocities change
    _, _, _, viol = cp.read_output(out, cd_t)
    assert any("timing_only" in v for v in viol)
    cd_v = cp.prepare_cell(by_id["f2_velocity_only"])
    _, _, _, viol = cp.read_output(out, cd_v)
    assert any("velocity_only" in v for v in viol)


def test_identity_check(cells, profile, tmp_path):
    """Identity cells: the reference output passes; a tampered file fails."""
    by_id = {c.id: c for c in cells}
    cell = by_id["f1_intensity00"]
    from wobblemidi.humanise import humanise
    out = tmp_path / "o.mid"
    humanise(cell.input_path, out, profile, seed=99, **cell.params)
    viol, warn = cp.check_identity(cell.input_path, out)
    assert viol == []

    mid = mido.MidiFile(str(out))
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                msg.velocity = min(127, msg.velocity + 1)
                break
    p = tmp_path / "tampered.mid"
    mid.save(str(p))
    viol, warn = cp.check_identity(cell.input_path, p)
    assert any("note events differ" in v for v in viol)


# ── scoring behaviour ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def f1_pools(cells, profile):
    """Two disjoint-seed 8-run pools on f1 (small, for mechanics only)."""
    by_id = {c.id: c for c in cells}
    cd = cp.prepare_cell(by_id["f1_default"])
    a = cp.generate_runs(cd, profile, [50_000 + i for i in range(8)])
    b = cp.generate_runs(cd, profile, [60_000 + i for i in range(8)])
    return cd, a, b


def test_score_pools_identical_pool_is_zero(f1_pools):
    """A pool scored against itself: every pooled distance is exactly 0 and
    every run-stat diff is exactly 0 — the floor of 'equivalent'."""
    cd, a, _ = f1_pools
    scores = cp.score_pools(cd, a, a)
    assert scores, "no metrics computed"
    for key, v in scores.items():
        if np.isfinite(v):
            assert v == pytest.approx(0.0, abs=1e-12), key


def test_score_pools_disjoint_seeds_small_but_nonzero(f1_pools):
    """Disjoint seed pools differ a little (null variance) but W1 distances stay
    in the low-ms / low-velocity range — the runner's whole premise."""
    cd, a, b = f1_pools
    scores = cp.score_pools(cd, a, b)
    assert scores["ALL|off_w1"] > 0
    assert scores["ALL|off_w1"] < 5.0        # ms; generous — null is ~0.2-1
    assert scores["ALL|veld_w1"] < 5.0
    assert 0 <= scores["ALL|off_ks"] <= 1


def test_score_pools_detects_gross_timing_scale(f1_pools):
    """A crude offset doubling must dwarf the null distances — the mechanism the
    calibrated thresholds formalise."""
    cd, a, b = f1_pools
    null = cp.score_pools(cd, a, b)
    import dataclasses
    scaled = [dataclasses.replace(r, off=r.off * 2.0) for r in b]
    bad = cp.score_pools(cd, a, scaled)
    assert bad["ALL|off_sigma_d"] > 4 * max(null["ALL|off_sigma_d"], 0.05)
    assert bad["ALL|off_w1"] > 2 * max(null["ALL|off_w1"], 0.05)


def test_timing_only_cell_scores_no_velocity_metrics(cells, profile):
    by_id = {c.id: c for c in cells}
    cd = cp.prepare_cell(by_id["f2_timing_only"])
    runs = cp.generate_runs(cd, profile, [50_000, 50_001])
    scores = cp.score_pools(cd, runs, runs)
    assert not any(k.endswith(("veld_w1", "wrole_sigma_d", "zjump_d", "v_lag1_d"))
                   for k in scores)
    assert "ALL|off_w1" in scores


# ── calibrated-verdict locks (need calibration/tier2_thresholds.json) ───────

THRESHOLDS = cp.THRESHOLDS_DEFAULT

needs_thresholds = pytest.mark.skipif(
    not THRESHOLDS.exists(),
    reason="locked thresholds not present (calibration not yet committed)",
)


@pytest.fixture(scope="module")
def locked_thresholds():
    return cp.load_thresholds(THRESHOLDS)


def _verdict_for(cells, profile, thresholds, cell_ids, candidate_pools):
    scores, violations = {}, {}
    by_id = {c.id: c for c in cells}
    for cid in cell_ids:
        cd = cp.prepare_cell(by_id[cid])
        ref = cp.generate_runs(
            cd, profile, [cp.REFERENCE_SEED_BASE + i for i in range(cp.K_RUNS)])
        runs_c = candidate_pools[cid](cd)
        viols = [v for r in runs_c for v in r.violations]
        if viols:
            violations[cid] = viols
        scores[cid] = cp.score_pools(cd, runs_c, ref)
    return cp.evaluate(scores, violations, thresholds, full_cell_set=False)


@needs_thresholds
def test_self_null_verdict_passes_on_ci_subset(cells, profile, locked_thresholds):
    """THE null lock: the reference engine at fresh seeds is a correct 'port' and
    must pass the locked bands on a fast cell subset (full self-null is a script
    run, not a CI test)."""
    subset = ["f1_default", "f4_default", cp.FULL_KIT_CELL_ID]
    pools = {
        cid: (lambda cd, base=4_242_000 + idx * 10_000: cp.generate_runs(
            cd, profile, [base + i for i in range(cp.K_RUNS)]))
        for idx, cid in enumerate(subset)
    }
    verdict = _verdict_for(cells, profile, locked_thresholds, subset, pools)
    assert verdict.passed, f"self-null failed: {verdict.failures}"


@needs_thresholds
def test_gross_mutant_verdict_fails(cells, profile, locked_thresholds):
    """Teeth check: offset samples ×1.5 (a gross port bug) must FAIL the locked
    bands on f1 alone. The perturbation is a test-local monkeypatch — the engine
    itself is untouched (golden vectors prove it in this same suite)."""
    import wobblemidi.humanise as H
    orig = H._sample_bucket

    def scaled(bucket):
        off, vd = orig(bucket)
        return off * 1.5, vd

    def make_pool(cd):
        H._sample_bucket = scaled
        try:
            return cp.generate_runs(cd, profile,
                                    [4_343_000 + i for i in range(cp.K_RUNS)],
                                    strict=False)
        finally:
            H._sample_bucket = orig

    verdict = _verdict_for(cells, profile, locked_thresholds,
                           ["f1_default"], {"f1_default": make_pool})
    assert not verdict.passed
    assert any("off_" in f for f in verdict.failures)
