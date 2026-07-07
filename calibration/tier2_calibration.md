# Tier 2 calibration record — port-vs-reference equivalence bands

_Locked 2026-07-07 after threshold sign-off (margin 1.62 approved by owner).
This file is the durable evidence behind `calibration/tier2_thresholds.json`,
which `scripts/compare_port.py verify` gates against. Regenerating the
thresholds is a deliberate act: rerun `scripts/calibrate_tier2.py`, take the
new evidence through the same sign-off, and update this record in the same
commit. Full numeric detail: `calibration/tier2_calibration_full.json`._

Environment: macOS arm64 / CPython 3.14.4 / numpy 2.5.1 / scipy 1.18.0, engine
at rev `5fdfb55`, profile sha `6ab0cf54…` (the golden-manifest profile).

## What was calibrated

`compare_port.py` grades a port by comparing pooled output distributions
against the reference engine on 23 distributional cells — the golden vectors
(deduplicated: `f1_cli`, `f1_seed7` skipped) plus the runner-owned full-kit
coverage fixture `tests/tier2/t2_full_kit.mid` (240 hits, all 22 TD-11 notes;
the golden inputs only cover notes 36/38/42/46/49) — at **K = 32
independent-seed runs per pool**. Identity cells (`f9_empty_default`,
`f1_intensity00`) are structural-equality checks and need no calibration.

**Gate structure (799 comparisons + 13 aggregates):**

- **Empirical gates (771):** per (cell, instrument-group-or-ALL, metric) —
  offset-from-input W1/KS/|Δmean|/|Δσ|, velocity-delta W1/|Δmean|/|Δσ|,
  per-run timing lag-1 / kick-velocity lag-1 / within-(position, role) σ /
  zero-jump mass diffs, plus per-note off/veld W1 on the full-kit cell.
  Threshold = null q99.5 × margin 1.62.
- **Degenerate absolute gates (28):** where the reference behaviour is
  deterministic the null is exactly zero across all 400 replicates —
  rigid-cluster gap deviation (`cdev_w1`), note_off placement (`offd_w1`),
  same-slot chord gap σ (`xgap_sigma_d`). These are **0.5 ms absolute
  tolerances**, margin-independent. (Round-1 finding: dropping zero-variance
  comparisons as "ungateable" left coupling mutants detected at ratios
  1.04–1.23; as structural gates they detect at 1.9–13×.)
- **Aggregate family gates (13):** mean null-normalised z per metric family
  across all gated cells — catches a port slightly wrong *everywhere* while
  under threshold in each cell individually.
- **Structural contracts (ungated, hard fail):** hit alignment, melodic-channel
  passthrough, tempo/time-signature preservation, timing_only/velocity_only
  invariants, velocity range, identity cells.

## Null model and threshold derivation

The runner compares a candidate pool against a **pinned** reference pool
(seeds 777000+0…31). Calibration mirrors that exactly: master pool of 256
reference runs per cell; one null replicate = the runner's fixed pool A vs a
random disjoint 32-run subset of the remaining 224, same subset indices across
all cells (preserves the cross-cell correlation the aggregates need).
**400 replicates.**

**Margin selection — full-verdict false-failure control, not fiat.** Per
replicate, M = max over all 771 empirical gates of value/null_q99.5. Observed:
mean 1.10, q99.5 = 1.49, max = 1.62 — i.e. even the single worst gate ratio in
400 simulated correct-port verdicts never exceeded 1.62. The auto-derived
margin (q99.5(M)×1.05 = 1.5619) measured a 1/20 held-out false-failure rate
(one gate at ratio 1.0026); **margin locked at 1.62 = the observed null
max-ratio**, at which all 20 held-out verdicts pass.

**Stability evidence** (thresholds are estimates; these bound the estimation
noise): per-batch q99 spread across 4 disjoint 100-replicate batches — median
0.12–0.21 of null_q per family; full recomputation under 3 alternative fixed
reference pools — median relative null_q difference 0.10–0.21 per family
(individual small-sample keys up to ~1.0; the margin and the max-ratio
selection absorb this).

**Representative locked bands (physical units):** kit-wide offset W1 ≤ ~1.0 ms
(f2: 1.01), offset σ diff ≤ 0.50 ms, velocity-delta W1 ≤ ~0.9–1.0, timing
lag-1 diff ≤ 0.065, kick velocity lag-1 diff ≤ 0.13, chord/cluster/note_off
placement ≤ 0.5 ms absolute.

## Held-out full-verdict nulls (the measured false-failure rate)

20 fresh 32-run candidate pools (never-seen seeds), each taken through the
complete runner verdict against the locked thresholds: **20/20 PASS** at
margin 1.62. Worst gate ratio across all 20 × 799 comparisons: **0.967**;
worst aggregate ratio 0.844. (At the pre-sign-off margin 1.5619: 19/20, the
single failure a hair-width 1.0026.) A correct port that ever fails marginally
(ratio ≈ 1.0) may regenerate its pools with fresh seeds — seeds are free by
contract; a real defect shows 1.9–13× (below).

## Mutation battery (13 deliberate port-bug simulations)

All perturbations are monkeypatches / alternate profile objects / parameter
overrides inside `scripts/calibrate_tier2.py` only — the engine source is
never touched (`verify_golden.py` green at every commit proves it). Ratios
below are worst gate ratio at margin 1.62; DETECTED = full verdict FAIL.

| Mutant | Simulated port bug | Worst ratio | Detected by |
|---|---|---:|---|
| couple_off | 12 ms rigid-cluster coupling not ported | 12.7 | `cdev_w1` degenerate gates (f4 cells) |
| map_42_ride | mapping error on a core note (42→ride) | 5.8 | group off/veld dists, 20 cells |
| tier_flatten | relative tiering (B4) not ported | 5.1 | snare veld W1, wrole σ + 6 aggregates |
| debias_off | push/lean de-bias arithmetic skipped | 4.0 | off_mean on lean/push cells |
| vel_scale_up | velocity deltas ×1.15 | 3.3 | veld σ/W1 kit-wide |
| phivel_off | kick velocity AR clock not ported | 2.1 | kick `v_lag1_d` (t2, f3) |
| off_scale_dn | timing offsets ×0.90 | 2.0 | off σ/W1 kit-wide |
| couple_loose | chord residual ±1 ms → ±25 ms | 1.9 | `xgap_sigma_d` degenerate gates, veld σ |
| kde_bw_x1.5 | KDE bandwidth factor ×1.5 | 1.9 | veld/off σ (f2_fill, t2) |
| off_scale_up | timing offsets ×1.10 | 1.6 | off σ/W1 kit-wide |
| phi_nudge | timing drift phi +0.1 | 1.6 | `t_lag1_d` (t2, f2) + aggregate |
| map_22_edge | mapping error on edge note 22→open | 1.15 | per-note `n22\|veld_w1`, **t2 only** |
| **phivel_nudge** | **PHI_VEL 0.37→0.47 transcription error** | **0.61** | **NOT DETECTED — recorded limitation** |

**Safety margin summary:** null ceiling (worst held-out ratio) 0.967 vs
weakest detected mutant 1.15 (`map_22_edge`) — a thin but real gap on that one
mutant, and ≥ 1.6 for every other; the coupling and mapping-core mutants sit
5–13× above the bands. `map_22_edge` is detectable **only** because of the
full-kit fixture and its per-note gates (round-1 finding: at group level it
scored 1.05).

## Recorded limitations

1. **PHI_VEL ±0.1 transcription error is below the Tier 2(b) noise floor at
   K=32** (worst ratio 0.61; the effect on kick velocity lag-1 is ~0.085,
   comparable to the null spread of the estimator). Mitigations: the
   not-ported failure mode (`phivel_off`) IS detected at 2.1×; and Tier 2(a)'s
   harness envelope records kick velocity lag-1 vs human (+0.32), bounding how
   far the coefficient can drift before the other gate fails. **Escalation:
   if a port's kick `v_lag1_d` values sit suspiciously close to their bands,
   rerun both pools at higher K (e.g. 128) on `t2_full_kit`/`f3_default` only
   — estimator noise shrinks ~√K while a real coefficient error does not; the
   comparison is then informational (the locked bands are only valid at K=32)
   but cleanly separates the two cases.**
2. **Thresholds are valid only at K=32 pools against the pinned reference
   seeds** — that is the calibrated statistic. A different K requires
   recalibration.
3. **Low-hit cells (f5, f6, f9_single) carry honest-but-loose bands** (e.g.
   f9_single ALL off_w1 ≤ 2.8 ms) — they are meter-path/degenerate smoke
   tests; f2/f3/t2 carry the statistical power.
4. Per-note mapping coverage is closed by `t2_full_kit` for all 22 TD-11
   notes; mapping errors on notes **absent from GMD-derived buckets** would
   surface as fallback-chain changes and are covered by the same distributional
   gates, but no mutant explicitly exercised that path.

## Reproduction

```
python scripts/calibrate_tier2.py                  # full run, ~35 min
python scripts/calibrate_tier2.py --rescale-margin-only 1.62
python scripts/compare_port.py self-null           # full null verdict, ~1 min
pytest tests/test_compare_port.py                  # CI locks (subset self-null + gross mutant)
```
