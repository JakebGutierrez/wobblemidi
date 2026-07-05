# wobblemidi — Step 3 spec v2: velocity rebuild + validation harness

_Design spec, revised after Codex design review. Covers the module-13 velocity rebuild, the
supporting profile-schema changes, and the validation harness that gates it. Numbers come from
the GMD measurements in `wobblemidi_roadmap.md` — read that first._

## What changed from v1 (post-review)
- **B3 (static per-song feel-offset) removed from this rebuild.** It double-counts with the frozen
  phi=0.4 (extracting the lean implies phi≈0.21). B3 only makes sense packaged with a phi
  recalibration, which is gated on a clean ear test. Parked as a paired decision.
- **This rebuild is velocity-only (+ tiering).** Timing is untouched — it stays exactly as Step 1
  left it. We only change what we measured is broken.
- **Kick VelDrift is kick-only, no sharing to crash** (no measured cross-instrument *velocity*
  correlation supports sharing).
- **A2 gains two guardrails:** per-emitted-bucket de-bias to ~0, and sparse-cell shrinkage.
- **Harness gains contour-preservation metrics, multi-seed CIs, and a clean train-split baseline.**

## Locked decisions (audit within these; do not reopen)
Paradigm KEEP; target rock/pop; one batched rebuild; velocity fix = redefine `vel_delta` at build
time; AR(1) on velocity is kick-only; cross-instrument timing already handled by the Step-1
kit-wide clock; phi stays 0.4 (fast-tempo ear test was confounded, re-tested after this rebuild).
Parked: B3+phi recalibration, tight-curated profile variant, presets/intensity default,
tempo-aware phi.

---

## Build order (harness first)
1. **Build the validation harness** (Part C) and establish baselines: distance-to-human for
   (a) quantised input, (b) an old-schema profile rebuilt from the **train split only** (the clean
   gate), and (c) current shipped `rock.json` (product context only — it has seen the held-out
   takes, so it's context, not the gate).
2. **Do the batched profile rebuild** (Part A) + runtime changes (Part B).
3. **Re-measure.** Accept only against the Part C criteria vs the clean train-split baseline.

---

## Part A — build-time changes (`scripts/build_profiles.py`)

### A1. Tag each hit with its source take
Carry a take/file ID through `_build_pairs`. Unlocks A2 and the harness train/test split.

### A2. Redefine `vel_delta` (core fix) — with de-bias + shrinkage
- New definition: `vel_delta = velocity − mean_velocity_for_(take, grid_position, instrument)`.
  Strips accent structure (which lives in the contour the **user programs**) out of the sampled
  noise, leaving within-performance imperfection. Applied velocity σ roughly halves (30–39 → 15–21).
- **Guardrail 1 — bucket-level de-bias (required).** Residualising at the (take, position,
  instrument) level does NOT guarantee a zero mean at the *tier bucket* level the runtime samples
  from (soft/medium/hard can still carry non-zero residual means, which would re-introduce accent
  double-counting). So: after residualising, compute, assert, and store `vel_delta_mean` per
  **emitted** bucket, and de-bias each emitted bucket to ~0. Equivalent alternative: compute the
  contour baseline at the exact conditioning level used for sampling.
- **Guardrail 2 — sparse-cell shrinkage (required).** `n=1` `(take, position, instrument)` cells
  produce a residual of exactly 0 (fake zero-noise crashes/fills). Shrink sparse-cell means toward
  a broader take/instrument or global-position mean before residualising. Define the shrinkage
  threshold and target level.
- Store `vel_sigma_within` per bucket in `_meta`.

### A3. (Optional, capture-only) record the per-take timing lean distribution in `_meta`
As reference data for the future B3+phi decision. Do **not** change stored offsets or wire anything
at runtime this round. Skip if it complicates the rebuild.

### A4. Storage contract unchanged
Keep `[[offset, vel_delta], …]` pairs (preserves the 2D-KDE upgrade path). Only `_meta` grows.

---

## Part B — runtime changes (`wobblemidi/humanise.py`)

### B1. Velocity: small residual on user contour
`new_vel = msg.velocity + sampled_vel_delta * intensity` — shape unchanged, but the sampled delta
is now the small A2 residual, so it perturbs the user's dynamics instead of overwriting accent roles.

### B2. Kick-only velocity AR(1) — no cross-instrument sharing
Add a `VelDrift` AR(1) clock (mirroring `GrooveDrift`) for **kick only** (kick lag-1 autocorr +0.32;
snare/hat/ride ~white → i.i.d.). **Do not** share kick velocity movement to same-tick crash/snare —
each instrument keeps its own draw. Calibrate `phi_vel` by extending `scripts/calibrate_phi.py` to
velocity lag-1 autocorr (per take, de-meaned per (instrument, position)).

### B3. (removed — parked) static per-song feel-offset. Not in this rebuild.

### B4. Relative velocity tiering (fixes the tier-collapse HIGH finding)
Map each hit's velocity to a tier by its **percentile within the user's own per-instrument track
distribution**, then look up the GMD bucket for that tier. This also fixes mis-routed timing (tier
selects the timing bucket too). Rules (from review):
- Use relative tiering only with enough evidence: `n ≥ 8` hits for that instrument **and**
  `p90 − p10 ≥ 12` velocity units.
- Two-cluster (ghost/accent) parts: map the low cluster → soft, high → hard; do **not** force a
  bogus medium.
- All-one-velocity or tiny parts: fall back to GMD-absolute thresholds. Preserve ties.

### B5. phi unchanged at 0.4. Kit-wide clock unchanged (Step 1).

---

## Part C — validation harness (build first; new script, e.g. `scripts/validate.py`)

### Method
1. Clean train/test split of GMD rock **by take**. Profiles for the gate are built from train only.
2. For each held-out human performance, build a "programmed" input: quantise timing to the 16th
   grid; coarsen velocity to a **4-level per-instrument palette per take** (O1) — a producer
   programs a contour, not micro-noise. Also run 2-level, 8-level, and flat as sensitivity/stress
   cases, not as the gate.
3. Run wobblemidi on the programmed input.
4. Compare against the real human original: programmed input vs current-schema (train-split) vs
   rebuilt-schema output.

### Metrics (per instrument, over held-out tracks)
- **Distribution match:** offset distribution vs grid (Wasserstein/KS; match σ and mean); velocity
  within-position σ; adjacent-hit velocity-jump distribution (the metric that exposed the
  machine-gun problem — hats jumped mean ~38 pre-fix).
- **Contour preservation (added post-review — the anti-gaming metrics):** per-(instrument,
  grid-position) mean-velocity error; Spearman rank correlation of output velocity contour against
  the coarsened input (and original) contour. These catch "right distribution, wrong hits landed
  loud" — which the distribution metrics alone can miss.
- **Correlation:** lag-1 autocorr, timing and velocity (approach human values, don't overshoot).
- **Cross-instrument:** same-slot gap σ (near human's ~15–20 ms, not ~1 ms, not the pre-Step-1
  ~73 ms).

### Robustness
Run **multiple fixed seeds** and report confidence intervals — a single stochastic run is too noisy
to gate on.

### Acceptance criterion
Accept the rebuild only if, on held-out data, distance-to-human is lower than **both** the quantised
input and the **clean train-split old-schema** profile, on velocity fine-structure **and**
contour-preservation, without regressing the timing metrics (which shouldn't move — timing is
untouched). This is also the empirical form of revisit-trigger #3: if the rebuild can't beat the
baseline here, that's the signal to reconsider the paradigm, not to keep tuning.

---

## Test locks to retire / replace (O5)
Exact-constant locks that will fight this: the `(1-RESIDUAL_SHARE)*phi` autocorr test and the
AR-recursion-constant test. Because B3 is dropped, the push / default-on-grid / variance-preservation
/ sample-stream-invariance tests and `calibrate_phi.py`'s formula are **not** disturbed by timing
changes this round — keep them. Replace the two locks with invariants: phi=0 bypass, autocorr
monotone in phi, total-variance budget respected, reproducibility across seeds, and the
velocity-de-bias contract (per-bucket residual mean ≈ 0).

## Not in this spec (deferred)
B3 + phi recalibration (paired, post clean ear test); tight-curated variant; presets/intensity
default; profile rounding; pandas→optional; pyproject/CI polish.
