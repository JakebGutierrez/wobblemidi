"""Calibrate the Tier 2 runner: null variance, thresholds, mutation battery.

Produces the empirical evidence that locks compare_port.py's equivalence bands
(design-note-approved procedure, 2026-07-07):

  calibration/tier2_thresholds.json        — the gates compare_port.py verifies against
  calibration/tier2_calibration_full.json  — full numeric detail

(calibration/tier2_calibration.md, the human-readable evidence record, is
written from these at threshold sign-off and updated in the same commit as
any regeneration.)

Null model — matches the runner exactly. The runner compares a candidate pool
against a FIXED reference pool (seeds REFERENCE_SEED_BASE+0..K-1). Calibration
builds a master pool of R reference runs per cell (seeds BASE+0..R-1, so the
runner's pool is master[0:K]) and a null replicate is: fixed A = master[0:K] vs
a random disjoint K-subset of master[K:R]. The same subset indices are used
across all cells within a replicate — one 'engine' ran one seed set on every
cell — preserving the cross-cell correlation the aggregate z thresholds need.

Thresholds: the QUANTILE (default 0.995) of each comparison's null distribution
× a margin derived from the null max-ratio distribution (per replicate,
M = max over gates of value/null_q; margin = q(M) × buffer — full-verdict
false-failure control across ~800 correlated gates; final value fixed at
threshold sign-off). Deterministic reference behaviours (all-zero nulls:
rigid-cluster gaps, fixed note_offs, tick-tight chords) become absolute
tolerance gates instead of being dropped. Raw null max is recorded alongside,
and stability is demonstrated two ways: per-batch quantile spread across 4
disjoint replicate batches, and full recomputation under 3 alternative
fixed-A pools.

Mutation battery: deliberate engine perturbations applied as monkeypatches /
alternate profile objects / parameter overrides INSIDE THIS SCRIPT ONLY — the
engine source is never modified (verify_golden.py green proves it). Every mutant
must be flagged DIFFERENT at full-verdict level; an undetected mutant is a
recorded finding, never silently dropped.

Held-out full-verdict nulls: fresh candidate pools (never-seen seeds) evaluated
against the locked thresholds exactly as the runner would — the measured
false-failure rate at verdict level.

Usage:
    python scripts/calibrate_tier2.py            # full run (~15-20 min)
    python scripts/calibrate_tier2.py --quick    # reduced dev run (numbers not for locking)
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import json
import math
import platform
import subprocess
import sys
import warnings
from importlib.metadata import version as pkg_version
from pathlib import Path

import click
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from wobblemidi import humanise as humanise_mod
from wobblemidi import midi_utils as midi_utils_mod
from wobblemidi.humanise import load_profile

import compare_port as cp
from verify_golden import load_bundled_profile, load_manifest

CALIB_DIR = REPO_ROOT / "calibration"
SHIPPED_PROFILE = REPO_ROOT / "wobblemidi" / "profiles" / "rock.json"

MASTER_RUNS = 256           # reference master pool size per cell
N_REPLICATES = 400          # null replicates (fixed A vs random disjoint B)
QUANTILE = 0.995
MARGIN_FLOOR = 1.3          # auto margin never goes below the design-note default
MARGIN_BUFFER = 1.05        # buffer over the null max-ratio quantile
MIN_FINITE_RATE = 0.9       # a comparison must be computable in >=90% of null reps
MIN_NULL_Q = 1e-9           # below this the null is (near-)deterministic

# Deterministic reference behaviours produce all-zero nulls (rigid clusters are
# exactly rigid; note_offs are fixed; same-tick chords land tick-tight). Those
# comparisons must not be dropped as "no variance" — they become ABSOLUTE
# tolerance gates (ms), margin-independent: the port must reproduce the
# deterministic behaviour within half a coupled-residual cap.
# (Calibration round 1 finding: dropping them left couple_off/couple_loose
# detected only at ratios 1.04-1.23.)
DEGENERATE_TOL_MS = {"cdev_w1": 0.5, "offd_w1": 0.5, "xgap_sigma_d": 0.5}
N_ALT_A = 3                 # alternative fixed-A pools for threshold stability
N_ALT_REPLICATES = 100
N_HELD_OUT = 20             # fresh full-verdict null runs against locked thresholds
MUTANT_SEED_BASE = 999_000
HELD_OUT_SEED_BASE = 1_500_000


# ---------------------------------------------------------------------------
# Mutation battery — monkeypatch/profile/param perturbations, NEVER engine edits
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_attr(obj, name, value):
    orig = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, orig)


@contextlib.contextmanager
def _patch_mapping(note: int, group: str):
    orig = midi_utils_mod.TD11_TO_GROUP[note]
    midi_utils_mod.TD11_TO_GROUP[note] = group   # same dict object humanise imported
    try:
        yield
    finally:
        midi_utils_mod.TD11_TO_GROUP[note] = orig


def _scaled_sampler(offset_factor: float = 1.0, veld_factor: float = 1.0):
    orig = humanise_mod._sample_bucket

    def patched(bucket):
        off, vd = orig(bucket)
        return off * offset_factor, vd * veld_factor

    return patched


def _kde_bw_profile(factor: float):
    """Fresh profile with every bucket KDE's bandwidth scaled by *factor*."""
    prof = load_profile(SHIPPED_PROFILE)
    for bucket in prof.buckets.values():
        if bucket.kde is not None:
            bucket.kde.set_bandwidth(bw_method=bucket.kde.factor * factor)
    return prof


def _nudge_phi(params: dict) -> dict:
    p = dict(params)
    if p.get("phi", 0.0) > 0.0:
        p["phi"] = min(0.95, p["phi"] + 0.1)
    return p


@dataclasses.dataclass
class Mutant:
    name: str
    description: str
    simulates: str
    patch: object = None            # contextmanager factory, or None
    profile_factory: object = None  # () -> LoadedProfile, or None
    params_transform: object = None # dict -> dict, or None


def build_mutants() -> list[Mutant]:
    import dataclasses as dc

    return [
        Mutant("off_scale_up", "timing offset samples ×1.10",
               "sample-path gain error ~ +10%",
               patch=lambda: _patch_attr(humanise_mod, "_sample_bucket",
                                         _scaled_sampler(offset_factor=1.10))),
        Mutant("off_scale_dn", "timing offset samples ×0.90",
               "sample-path gain error ~ -10%",
               patch=lambda: _patch_attr(humanise_mod, "_sample_bucket",
                                         _scaled_sampler(offset_factor=0.90))),
        Mutant("vel_scale_up", "velocity delta samples ×1.15",
               "velocity gain error ~ +15%",
               patch=lambda: _patch_attr(humanise_mod, "_sample_bucket",
                                         _scaled_sampler(veld_factor=1.15))),
        Mutant("phi_nudge", "phi +0.1 (cells with phi>0)",
               "wrong AR(1) drift coefficient in the timing clock",
               params_transform=_nudge_phi),
        Mutant("phivel_nudge", "PHI_VEL 0.37 → 0.47",
               "kick velocity drift coefficient transcription error",
               patch=lambda: _patch_attr(humanise_mod, "PHI_VEL", 0.47)),
        Mutant("phivel_off", "PHI_VEL 0.37 → 0.0 (kick velocity drift disabled)",
               "kick velocity AR clock not ported at all",
               patch=lambda: _patch_attr(humanise_mod, "PHI_VEL", 0.0)),
        Mutant("tier_flatten", "relative tiering disabled (absolute fallback)",
               "per-file relative tier thresholds (B4) not ported",
               patch=lambda: _patch_attr(humanise_mod, "_file_tier_thresholds",
                                         lambda v, absolute: absolute)),
        Mutant("kde_bw_x1.5", "KDE bandwidth ×1.5 on every bucket",
               "wrong bandwidth factor in the KDE reimplementation",
               profile_factory=lambda: _kde_bw_profile(1.5)),
        Mutant("couple_off", "COUPLE_WINDOW_MS → -1 (windowed clusters disabled)",
               "12 ms rigid-cluster coupling not ported",
               patch=lambda: _patch_attr(humanise_mod, "COUPLE_WINDOW_MS", -1.0)),
        Mutant("couple_loose", "COUPLED_RESIDUAL_MS 1 → 25",
               "same-tick chord coupling residual mis-ported (loose chords)",
               patch=lambda: _patch_attr(humanise_mod, "COUPLED_RESIDUAL_MS", 25.0)),
        Mutant("debias_off", "bucket_offset_means dropped (de-bias skipped)",
               "push/lean de-bias arithmetic not ported",
               profile_factory=lambda: dc.replace(load_profile(SHIPPED_PROFILE),
                                                  bucket_offset_means={})),
        Mutant("map_42_ride", "TD-11 note 42 (closed hat) → ride bucket",
               "instrument-mapping error on a core note",
               patch=lambda: _patch_mapping(42, "ride")),
        Mutant("map_22_edge", "TD-11 note 22 (HH closed edge) → hihat_open bucket",
               "instrument-mapping error on an edge-variant note "
               "(only the full-kit fixture contains note 22)",
               patch=lambda: _patch_mapping(22, "hihat_open")),
    ]


# ---------------------------------------------------------------------------
# Calibration core
# ---------------------------------------------------------------------------

def _quantile(vals: np.ndarray, q: float) -> float:
    return float(np.quantile(vals, q)) if len(vals) else float("nan")


def null_replicate_scores(cds, master, a_idx, b_idx) -> dict[str, dict[str, float]]:
    return {
        cd.cell.id: cp.score_pools(
            cd,
            [master[cd.cell.id][i] for i in b_idx],     # candidate side
            [master[cd.cell.id][i] for i in a_idx],     # fixed reference side
        )
        for cd in cds
    }


def compute_thresholds(null_scores: list[dict], quantile: float,
                       margin: float | None) -> tuple[dict, dict, dict]:
    """(comparisons, per-key null arrays, margin_info) from replicate scores.

    margin=None → auto: per replicate, M_r = max over empirical gates of
    value / null_q; margin = max(MARGIN_FLOOR, q(M) x MARGIN_BUFFER). This
    targets full-verdict false-failure control directly (multiplicity across
    ~800 correlated gates), instead of a fiat per-gate factor. Degenerate
    (all-zero-null) gates take absolute tolerances and ignore the margin.
    """
    keys: dict[str, list[float]] = {}
    for rep in null_scores:
        for cell_id, sc in rep.items():
            for gm, v in sc.items():
                keys.setdefault(f"{cell_id}|{gm}", []).append(v)

    comparisons: dict[str, dict] = {}
    null_values: dict[str, np.ndarray] = {}
    n_reps = len(null_scores)
    for key, vals in keys.items():
        arr = np.array(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        finite_rate = len(finite) / n_reps
        null_values[key] = finite
        if finite_rate < MIN_FINITE_RATE:
            continue
        q = _quantile(finite, quantile)
        if not math.isfinite(q):
            continue
        metric = key.split("|")[2]
        spec = {
            "null_q": q,
            "null_max": float(finite.max()),
            "null_mean": float(finite.mean()),
            "null_std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            "finite_rate": round(finite_rate, 4),
        }
        if q <= MIN_NULL_Q:
            if metric in DEGENERATE_TOL_MS:
                spec["threshold"] = max(DEGENERATE_TOL_MS[metric],
                                        spec["null_max"] * 1.5)
                spec["degenerate"] = True
                comparisons[key] = spec
            continue    # non-degenerate zero-variance: structural, not gated
        comparisons[key] = spec

    # auto margin from the null max-ratio distribution over empirical gates
    m_vals = []
    for rep in null_scores:
        ratios = []
        for key, spec in comparisons.items():
            if spec.get("degenerate"):
                continue
            cell_id, group, metric = key.split("|")
            v = rep.get(cell_id, {}).get(f"{group}|{metric}", float("nan"))
            if math.isfinite(v):
                ratios.append(v / spec["null_q"])
        if ratios:
            m_vals.append(max(ratios))
    m_arr = np.array(m_vals)
    margin_info = {
        "mode": "auto" if margin is None else "manual",
        "M_mean": float(m_arr.mean()),
        "M_q": _quantile(m_arr, quantile),
        "M_max": float(m_arr.max()),
    }
    if margin is None:
        margin = max(MARGIN_FLOOR, margin_info["M_q"] * MARGIN_BUFFER)
    margin_info["margin"] = round(float(margin), 4)

    for spec in comparisons.values():
        if not spec.get("degenerate"):
            spec["threshold"] = spec["null_q"] * margin_info["margin"]
    return comparisons, null_values, margin_info


def compute_aggregates(null_scores: list[dict], comparisons: dict,
                       quantile: float, margin: float) -> tuple[dict, dict]:
    """Per-metric-family mean-z thresholds over the null replicates."""
    fam_reps: dict[str, list[float]] = {}
    for rep in null_scores:
        fam_terms: dict[str, list[float]] = {}
        for key, spec in comparisons.items():
            if spec["null_std"] <= 0:
                continue
            cell_id, group, metric = key.split("|")
            v = rep.get(cell_id, {}).get(f"{group}|{metric}", float("nan"))
            if math.isfinite(v):
                fam_terms.setdefault(metric, []).append(
                    (v - spec["null_mean"]) / spec["null_std"])
        for m, terms in fam_terms.items():
            fam_reps.setdefault(m, []).append(float(np.mean(terms)))

    aggregates, agg_null = {}, {}
    for m, zs in fam_reps.items():
        arr = np.array(zs)
        q = _quantile(arr, quantile)
        aggregates[m] = {
            "threshold": q * margin,
            "null_q": q,
            "null_max": float(arr.max()),
            "n_replicates": len(arr),
        }
        agg_null[m] = arr
    return aggregates, agg_null


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def evaluate_candidate(cds, master, candidate_runs, thresholds) -> cp.Verdict:
    scores, violations = {}, {}
    ref = {cd.cell.id: master[cd.cell.id][:cp.K_RUNS] for cd in cds}
    for cd in cds:
        runs_c = candidate_runs[cd.cell.id]
        viols = [v for r in runs_c for v in r.violations]
        if viols:
            violations[cd.cell.id] = viols
        scores[cd.cell.id] = cp.score_pools(cd, runs_c, ref[cd.cell.id])
    return cp.evaluate(scores, violations, thresholds, full_cell_set=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--quick", is_flag=True,
              help="Reduced sizes for a fast dev pass — NOT for locking thresholds.")
@click.option("--margin", type=float, default=None,
              help="Manual safety factor over the null quantile. Default: auto — "
                   "derived from the null max-ratio distribution (full-verdict "
                   "false-failure control); final value fixed at sign-off.")
@click.option("--skip-held-out", is_flag=True, help="Skip the held-out verdict nulls.")
@click.option("--rescale-margin-only", type=float, default=None,
              help="No recomputation: rescale the existing thresholds file to this "
                   "margin (thresholds are null_q x margin, so a margin change at "
                   "sign-off needs no rerun). Mutant/held-out ratios in the full "
                   "record rescale as ratio x old_margin / new_margin.")
def main(quick: bool, margin: float, skip_held_out: bool,
         rescale_margin_only: float | None) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if rescale_margin_only is not None:
        thr_path = CALIB_DIR / "tier2_thresholds.json"
        doc = json.loads(thr_path.read_text())
        old = doc["_meta"]["margin"]
        for spec in doc["comparisons"].values():
            if spec.get("degenerate"):
                continue    # absolute tolerance gates do not scale with margin
            spec["threshold"] = spec["null_q"] * rescale_margin_only
        for spec in doc["aggregates"].values():
            spec["threshold"] = spec["null_q"] * rescale_margin_only
        doc["_meta"]["margin"] = rescale_margin_only
        doc["_meta"]["description"] = doc["_meta"]["description"].replace(
            f"margin {old}", f"margin {rescale_margin_only}")
        thr_path.write_text(json.dumps(doc, indent=1) + "\n")
        click.echo(f"rescaled {thr_path}: margin {old} -> {rescale_margin_only}")
        return
    rng = np.random.RandomState(20260707)

    master_runs = 96 if quick else MASTER_RUNS
    n_reps = 60 if quick else N_REPLICATES
    n_alt_reps = 20 if quick else N_ALT_REPLICATES
    n_held = 3 if quick else N_HELD_OUT

    manifest = load_manifest()
    profile, profile_sha = load_bundled_profile()
    cp.startup_selfcheck(profile, profile_sha, manifest)
    cells = [c for c in cp.load_cells(manifest) if c.kind == "dist"]
    click.echo(f"Preparing {len(cells)} distributional cells "
               f"(identity cells are structural — not calibrated)…")
    cds = [cp.prepare_cell(c) for c in cells]

    # -- master reference pool ------------------------------------------------
    click.echo(f"Master pool: {master_runs} reference runs per cell…")
    master: dict[str, list] = {}
    for cd in cds:
        seeds = [cp.REFERENCE_SEED_BASE + i for i in range(master_runs)]
        master[cd.cell.id] = cp.generate_runs(cd, profile, seeds)
        click.echo(f"  {cd.cell.id}: {len(cd.notes)} hits × {master_runs}")

    # -- null replicates: fixed A = master[0:K] vs random disjoint B ----------
    K = cp.K_RUNS
    click.echo(f"Null replicates: {n_reps} × (fixed A[0:{K}] vs random "
               f"B ⊂ [{K}:{master_runs}])…")
    a_idx = list(range(K))
    null_scores = []
    for _ in range(n_reps):
        b_idx = list(rng.choice(np.arange(K, master_runs), size=K, replace=False))
        null_scores.append(null_replicate_scores(cds, master, a_idx, b_idx))

    comparisons, null_values, margin_info = compute_thresholds(
        null_scores, QUANTILE, margin)
    margin_locked = margin_info["margin"]
    aggregates, agg_null = compute_aggregates(null_scores, comparisons,
                                              QUANTILE, margin_locked)
    n_degen = sum(1 for s in comparisons.values() if s.get("degenerate"))
    click.echo(f"  gated comparisons: {len(comparisons)} "
               f"({n_degen} degenerate absolute gates)  "
               f"(+{len(aggregates)} aggregate families)")
    click.echo(f"  margin [{margin_info['mode']}]: {margin_locked}  "
               f"(null max-ratio M: mean {margin_info['M_mean']:.2f}, "
               f"q{QUANTILE} {margin_info['M_q']:.2f}, max {margin_info['M_max']:.2f})")

    # -- stability 1: per-batch quantile spread -------------------------------
    n_batches = 4
    batch = max(1, n_reps // n_batches)
    batch_spread: dict[str, list[float]] = {}
    for key in comparisons:
        qs = []
        for bi in range(n_batches):
            vals = []
            for rep in null_scores[bi * batch:(bi + 1) * batch]:
                cell_id, group, metric = key.split("|")
                v = rep.get(cell_id, {}).get(f"{group}|{metric}", float("nan"))
                if math.isfinite(v):
                    vals.append(v)
            if vals:
                qs.append(_quantile(np.array(vals), 0.99))
        if len(qs) == n_batches and comparisons[key]["null_q"] > 0:
            spread = (max(qs) - min(qs)) / comparisons[key]["null_q"]
            batch_spread.setdefault(key.split("|")[2], []).append(spread)

    # -- stability 2: alternative fixed-A pools -------------------------------
    n_alt_a = min(N_ALT_A, master_runs // K - 1)
    click.echo(f"Stability: thresholds under {n_alt_a} alternative fixed-A pools…")
    alt_rel_diff: dict[str, list[float]] = {}
    for ai in range(n_alt_a):
        lo = K * (ai + 1)
        alt_a = list(range(lo, lo + K))
        rest = [i for i in range(master_runs) if i not in set(alt_a)]
        alt_scores = []
        for _ in range(n_alt_reps):
            b_idx = list(rng.choice(rest, size=K, replace=False))
            alt_scores.append(null_replicate_scores(cds, master, alt_a, b_idx))
        alt_comps, _, _ = compute_thresholds(alt_scores, QUANTILE, 1.0)
        for key, spec in comparisons.items():
            if key in alt_comps and spec["null_q"] > 0:
                rel = abs(alt_comps[key]["null_q"] - spec["null_q"]) / spec["null_q"]
                alt_rel_diff.setdefault(key.split("|")[2], []).append(rel)

    thresholds_doc = {
        "_meta": {
            "description": (
                "Tier 2 port-vs-reference equivalence bands. Calibrated empirically "
                "(scripts/calibrate_tier2.py): null = reference engine vs itself at "
                "disjoint seeds against the runner's pinned reference pool; "
                "threshold = null q{q} x margin {m}. Valid ONLY at K={k} runs per "
                "pool. See calibration/tier2_calibration.md."
            ).format(q=QUANTILE, m=margin_locked, k=K),
            "generated_at": datetime.date.today().isoformat(),
            "generated_at_rev": _git_rev(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": pkg_version("numpy"),
            "scipy": pkg_version("scipy"),
            "profile_sha256": profile_sha,
            "full_kit_fixture_sha256": cp._sha256_bytes(cp.full_kit_bytes()),
            "K_runs": K,
            "reference_seed_base": cp.REFERENCE_SEED_BASE,
            "master_runs": master_runs,
            "n_replicates": n_reps,
            "quantile": QUANTILE,
            "margin": margin_locked,
            "margin_selection": margin_info,
            "quick_mode": quick,
            "cells": [cd.cell.id for cd in cds],
        },
        "comparisons": comparisons,
        "aggregates": aggregates,
    }

    # -- mutation battery ------------------------------------------------------
    click.echo("Mutation battery…")
    mutant_results = []
    for mi, mut in enumerate(build_mutants()):
        seeds = [MUTANT_SEED_BASE + mi * 1000 + i for i in range(K)]
        prof_m = mut.profile_factory() if mut.profile_factory else profile
        ctx = mut.patch() if mut.patch else contextlib.nullcontext()
        cand = {}
        with ctx:
            for cd in cds:
                params = (mut.params_transform(cd.cell.params)
                          if mut.params_transform else None)
                cand[cd.cell.id] = cp.generate_runs(cd, prof_m, seeds,
                                                    params=params, strict=False)
        verdict = evaluate_candidate(cds, master, cand, thresholds_doc)
        tripped = sorted([r for r in verdict.rows if math.isfinite(r[3]) and r[3] > 1.0],
                         key=lambda r: -r[3])
        agg_tripped = [m for m, a in verdict.aggregates.items()
                       if a["z"] > a["threshold"]]
        detected = not verdict.passed
        mutant_results.append({
            "name": mut.name,
            "description": mut.description,
            "simulates": mut.simulates,
            "detected": detected,
            "n_tripped": len(tripped),
            "n_structural": sum(1 for f in verdict.failures if f.startswith("[structural]")),
            "worst_ratio": tripped[0][3] if tripped else (
                max((r[3] for r in verdict.rows if math.isfinite(r[3])), default=float("nan"))),
            "worst_key": tripped[0][0] if tripped else None,
            "top_tripped": [(k, round(r, 2)) for k, _v, _t, r in tripped[:8]],
            "tripped_cells": sorted({k.split("|")[0] for k, *_ in tripped}),
            "agg_tripped": agg_tripped,
        })
        click.echo(f"  {mut.name:<16} {'DETECTED' if detected else 'NOT DETECTED  ← FINDING'}"
                   f"  (tripped {len(tripped)}, worst ratio "
                   f"{mutant_results[-1]['worst_ratio']:.2f})")

    # -- held-out full-verdict nulls -------------------------------------------
    held_out = []
    if not skip_held_out:
        click.echo(f"Held-out full-verdict nulls: {n_held} fresh pools…")
        for j in range(n_held):
            seeds = [HELD_OUT_SEED_BASE + j * 10_000 + i for i in range(K)]
            cand = {cd.cell.id: cp.generate_runs(cd, profile, seeds) for cd in cds}
            verdict = evaluate_candidate(cds, master, cand, thresholds_doc)
            worst = max((r[3] for r in verdict.rows if math.isfinite(r[3])),
                        default=float("nan"))
            worst_agg = max(
                ((a["z"] / a["threshold"]) if a["threshold"] > 0 else float("nan")
                 for a in verdict.aggregates.values()), default=float("nan"))
            held_out.append({
                "passed": verdict.passed,
                "n_failures": len(verdict.failures),
                "worst_ratio": worst,
                "worst_agg_ratio": worst_agg,
                "failures": verdict.failures[:5],
            })
            click.echo(f"  held-out #{j + 1}: "
                       f"{'PASS' if verdict.passed else 'FAIL ← false failure'} "
                       f"(worst ratio {worst:.2f})")

    # -- write artifacts ---------------------------------------------------------
    CALIB_DIR.mkdir(exist_ok=True)
    thr_path = CALIB_DIR / "tier2_thresholds.json"
    thr_path.write_text(json.dumps(thresholds_doc, indent=1) + "\n")

    full = {
        "meta": thresholds_doc["_meta"],
        "null_summary_by_family": {
            m: {
                "n_keys": len([k for k in comparisons if k.split("|")[2] == m]),
                "batch_q99_spread_median": float(np.median(batch_spread.get(m, [float("nan")]))),
                "batch_q99_spread_max": float(np.max(batch_spread.get(m, [float("nan")]))),
                "alt_A_relq_median": float(np.median(alt_rel_diff.get(m, [float("nan")]))),
                "alt_A_relq_max": float(np.max(alt_rel_diff.get(m, [float("nan")]))),
            }
            for m in sorted({k.split("|")[2] for k in comparisons})
        },
        "mutants": mutant_results,
        "held_out": held_out,
    }
    full_path = CALIB_DIR / "tier2_calibration_full.json"
    full_path.write_text(json.dumps(cp_jsonable(full), indent=1) + "\n")

    click.echo(f"\nWrote {thr_path}")
    click.echo(f"Wrote {full_path}")
    click.echo("\nSummary:")
    click.echo(f"  gated comparisons: {len(comparisons)}; aggregates: {len(aggregates)}")
    n_det = sum(1 for m in mutant_results if m["detected"])
    click.echo(f"  mutants detected: {n_det}/{len(mutant_results)}")
    if held_out:
        n_pass = sum(1 for h in held_out if h["passed"])
        click.echo(f"  held-out verdict nulls: {n_pass}/{len(held_out)} PASS")
    if quick:
        click.echo("  QUICK MODE — numbers are for development only, not locking.")


def cp_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): cp_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [cp_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


if __name__ == "__main__":
    main()
