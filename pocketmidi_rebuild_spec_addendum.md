# pocketmidi — Step 3 spec addendum (post-Checkpoint-3)

_Addendum to `pocketmidi_rebuild_spec.md` v2. Two fixes surfaced by the Checkpoint-3 re-measure.
Read the Checkpoint-3 result first for context._

## Context — why this addendum exists
The rebuild beat the old-schema gate on all six velocity/contour metrics, decisively and outside
the CIs; timing was unmoved. Two things need fixing before acceptance:
1. **Snare regressed** vs the gate (alone among instruments) — a real design gap.
2. **The "beat-the-input" acceptance clause is discredited for velocity** — the yardstick, not the
   tool, is wrong. (This is O1 from the original review coming home.)

Neither reopens a locked decision. Scope stays velocity-only; timing untouched; phi 0.4.

---

## Fix 1 — snare: residualise within velocity tier (design fix, Codex-review the diff)

**Problem.** At a single grid position, snare alternates between ghost (soft) and backbeat (loud)
across bars. The A2 baseline — the `(take, grid_position, instrument)` cell mean — therefore lands
*between* the two roles, so the residual re-absorbs the ghost/backbeat accent structure it was
supposed to strip out. This is why snare's residual σ is the largest of all groups (20.7) and why
snare is the one instrument that got worse. It's the exact failure the rebuild removes everywhere
else, reappearing for the instrument where ghosts matter most.

**Fix.** For snare, condition the residual baseline on velocity **tier** as well:
`vel_delta = velocity − shrunk mean of (take, grid_position, instrument, tier)`, so ghost hits and
backbeat hits get separate baselines instead of one blended mean. Keep the existing A2 guardrails
(per-emitted-bucket de-bias to ~0; sparse-cell shrinkage toward a broader mean).

**Constraints / watch-items.**
- Adding the tier dimension shrinks cell sample counts — the sparse-cell shrinkage (SHRINKAGE_K)
  matters more here. Confirm snare tier cells don't fall to fake-zero; adjust the shrinkage target
  (e.g. fall back to the tier-agnostic `(take, position, instrument)` mean, then `(take,
  instrument)`) rather than to a global mean.
- Apply this **only where the data supports it.** Snare is the confirmed case. Do NOT blanket-apply
  tier-conditioning to all instruments — measure first whether hi-hat/ride/kick have the same
  ghost/accent-at-one-position structure; if they don't, tier-conditioning just thins their cells
  for no gain. Report per-instrument whether tier-conditioning helps before extending it.
- This uses the tier the profile is built under, which must be the same tier convention runtime
  uses (relative tiering, B4). Keep build/runtime tier definitions consistent, as with the
  de-meaning convention.

**Acceptance for this fix:** snare comes into line with the gate (no metric worse than the
old-schema gate outside CIs), and no other instrument regresses.

---

## Fix 2 — harness: replace the degenerate input baseline with an anti-robotic metric

**Problem.** The "programmed input" is a 4-level coarsening of the human take itself, so it inherits
the human's contour and aggregate velocity statistics by construction (its within-position σ ≈ human
17, contour MAE ≈ pure coarsening error). Any humaniser that adds life moves those aggregates *away*
from that artificially-clean starting point, so "beat the input on distance-to-human" is structurally
unpassable for velocity — the input is a flattened photocopy of the answer. (Mirror of the timing
side, where the input is σ=0 grid and obviously robotic, but the distance metric can't express it.)

**Fix — make the yardstick STRICTER about robotic-ness, not looser.** The legitimate reason to change
this clause is that the input is a photocopy of the answer; the illegitimate reason would be to make
the rebuild pass. So the replacement must *tighten* what "robotic" means, not relax the bar:
- Add a metric that isolates what's actually wrong with the coarsened input: **zero micro-jump mass**
  (identical repeated velocities → an adjacent-hit velocity-delta distribution with a large spike at
  0) and **no within-role spread** (velocity variance within a fixed (position, role) is ~0). Measure
  the mass of |adjacent velocity delta| = 0 (or within ±1), and the within-(position, tier) velocity
  σ; the coarsened input should score badly (high zero-mass, ~0 within-role σ) and a good humaniser
  should score near human on both.
- Reframe the acceptance clause: replace "beat the quantised input on distance-to-human" with "beat
  the quantised input on the anti-robotic metrics (zero-jump mass, within-role spread) **and** beat
  the old-schema gate on distance-to-human." The gate clause is unchanged and already passed; the
  new clause is what the input should now legitimately lose.

**Acceptance for this fix:** on the anti-robotic metrics the coarsened input scores clearly worse
than human and worse than the rebuilt output, confirming the metric captures the robotic quality the
distance metrics missed. Sanity-check it on the flat-input stress case too (flat should score worst).

**Process note.** Fix 2 is a change to the measuring tool, so validate it the way Checkpoint 1 was
validated — by watching it behave on known cases (coarsened, flat, human) — not by Codex review.
Fix 1 is real engine/build logic and gets a Codex diff review.

---

## Order
1. Build Fix 2 first (the corrected harness), so Fix 1 is measured against a fair yardstick.
2. Build Fix 1 (snare tier-residualisation). Codex-review that diff.
3. Re-measure: snare in line with gate, no other instrument regressed, timing unmoved, and the
   rebuilt output beats the gate on distance-to-human AND beats the input on the anti-robotic
   metrics. That's acceptance.
4. On acceptance → ship step (already recorded): rebuild shipped `rock.json` from the FULL GMD
   (train + test) with the v2 builder, add the schema_version-2 regression test.

## Still not in scope (unchanged)
B3 + phi recalibration; tight-curated variant; presets/intensity default; packaging/CI polish.
