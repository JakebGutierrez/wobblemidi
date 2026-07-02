# pocketmidi — plan & decisions

_Last updated: 2026-07-02. Working doc: locked decisions, parked backlog, and the build roadmap. Update it as things move._

## Status at a glance
- **Step 1 — engine fixes:** ✓ done (2 commits, 186 tests, Codex-reviewed clean).
- **Step 2 — ear test:** ✓ done → confounded; tempo-aware phi parked.
- **Step 3 — rebuild design:** ✓ done → `pocketmidi_rebuild_spec.md` v2, Codex design-reviewed.
- **Step 4 — build (harness → rebuild → re-measure):** ← next.

---

## Locked decisions

- **Paradigm: KEEP.** Statistical delta sampling + AR(1) groove drift is the right frame and the product's reason to exist (weights-free, dependency-light, note-preserving, interpretable knobs). A learned / GrooVAE-style rewrite was evaluated and rejected — it would be a solo-maintained clone competing with Magenta Studio on its own turf. Improvements are extensions *within* this paradigm.
- **Target genres: rock + pop now.** Funk / jazz / metal are explicit *later* scope, not current weaknesses. Swing genres are the paradigm's real boundary (see revisit triggers).
- **phi default 0.5 → 0.4.** Shipped in Step 1. Fast-tempo ear test came back confounded (velocity + timing spread dominate, not phi) → tempo-aware phi parked, re-test post-rebuild.
- **Module 13 (velocity rebuild).** Redefine `vel_delta` at build time as a residual against the `(take, grid-position, instrument)` mean (not the bucket median) — stops sampling accents as noise, roughly halves applied velocity spread. Guardrails: de-bias every emitted bucket to ~0 mean; shrink sparse cells. AR(1) is **kick-only** (kick autocorr +0.32; snare/hat/ride ~white → i.i.d.). Relative (user-percentile) velocity tiering. Static per-song feel-offset (B3) **removed and parked** — double-counts with phi=0.4. Full detail in `pocketmidi_rebuild_spec.md` (v2). NOT "AR on all velocities."
- **Cross-instrument timing: no new mechanism.** Adjacent cross-instrument correlation ≈ same-instrument, so one shared kit clock is correct — handled by the Step-1 kit-wide clock.
- **One batched profile rebuild** against one agreed schema spec — not two (per CLAUDE.md's "batch breaking profile changes").
- **Delivery shell is open** (CLI now, maybe GUI/plugin later). Keep the core light and embeddable regardless.

---

## Measured facts (from the GMD pass — don't re-derive these)

- Within-take timing σ 26–28 ms vs 29–31 ms total → between-take is only ~17%. Drummers genuinely play that loose against the grid. "Sloppy at intensity 1.0" is **taste, not a variance bug** — fix via default/presets.
- Take tightness varies ~2× (σ 17 ms at P10 vs 37 ms at P90) → a curated "tight" profile from the tighter half of takes is cheap and viable.
- Static per-take lean: std 11 ms across takes. ~Half the pooled slot correlation (r=0.318) is this static lean; true within-take wander r=0.179 → phi ≈ 0.21. (This is why B3 + phi 0.4 double-counts — see parked backlog.)
- Velocity lag-1 autocorr (within take, fixed position): kick +0.32, snare +0.15, hats +0.07, ride +0.09. Hats' scatter is **excessive magnitude, not missing correlation** — conditioning on `(take, position)` drops velocity σ from 30–39 to 15–21.
- Coupling is tighter than humans: same-slot GMD r=0.5–0.78 (kick+snare loosest 0.54, kick+crash tightest 0.78), σ≈15–20 ms. Current ±1 ms coupled residual is 10–20× tighter — a defensible taste call for produced music, worth an A/B (~5 ms) later.
- phi calibration: recommended ~0.374 overall; tempo split ~0.35–0.45 below 130 BPM, **0.09 above 130 BPM** (fast rock is nearly hit-to-hit independent).

---

## Work status — bugs & fixes

**Done (Step 1 — commits `5755df6` + `fa4a767`, Codex-reviewed clean):**
- Per-track groove clock flam → kit-wide clock (multi-track same-tick gap 73 ms → ≤1 tick).
- Drum-channel filter (default channel 10, `--all-channels` opt-out) — melodic parts no longer corrupted.
- Straight non-4/4 (incl. 3/4) accepted via `grid_pos=None`; 6/8-mixed still raises.
- phi default → 0.4; README / AGENTS / CLAUDE.md synced.

**Folded into the Step-3 spec (v2):**
- Velocity de-bias (A2 bucket-level guardrail).
- Relative velocity tiering (fixes the GMD-absolute tier-collapse).

**Queued — release polish (later):**
- Intensity default (→ 0.4 or presets), `--report` diagnostics, `pandas` → optional extra, profile rounding (~10 MB → ~5 MB), pyproject license/classifiers/URLs, CI version matrix.

---

## Parked backlog (write-down, don't touch now)

- **Validation harness** — promoted to a build step (Step 4). Listed here as the anchor for everything stochastic.
- Ear test done (2026-07-02) → **confounded**: both fast-tempo versions sounded bad because velocity + timing spread dominate, not phi. Tempo-aware phi stays parked; re-test after the velocity rebuild on freshly generated files.
- **B3 (static per-song feel-offset) + phi recalibration — paired decision, parked.** Extracting the take lean (σ≈11 ms) implies phi≈0.21, so B3 can't be added while phi stays 0.4 without double-counting. Revisit both together after the rebuild, gated on a clean ear test. Removed from the Step-3 rebuild.
- Tight-curated profile variant.
- Coupled-residual A/B (±1 ms vs ~5 ms).
- **Preserve-intent architecture** — add human residual *on top of* the user's input instead of flatten-to-grid-then-regenerate. Answers both the drums-in-isolation and iterative-use problems. Cheap first step: warn on / don't silently flatten off-grid input.
- Feed-the-song / groove-transfer ("play like this recording") — out of scope; it's revisit-trigger territory, not a knob.
- Known ceiling: humanising toward the *average* GMD drummer kills robotic but caps at "convincingly generic," not "distinctive feel." Fine for rock/pop backing.

### Revisit triggers for a learned approach (none close today; watch #3)
1. Swing genres (jazz/funk) come in-scope — per-slot stats can't represent phrase-dependent swing.
2. A small, permissive, embeddable open drum-humanisation model (GrooVAE-class, <50 MB, ONNX) appears to *call* rather than reimplement.
3. **The rebuild is built and A/B listening still says "programmed."** ← the empirical ceiling test (the harness is the objective form of this).
4. Users want "play like drummer X from this recording" (an embedding feature).

---

## Roadmap

**Step 1 — Safe engine fixes.** ✓ DONE. Channel filter, straight non-4/4, kit-wide clock, phi 0.4, docs. 186 tests green, Codex-reviewed clean.

**Step 2 — Ear test.** ✓ DONE → confounded (both fast versions bad; velocity + spread dominate, not phi). Tempo-aware phi parked, re-test post-rebuild.

**Step 3 — Rebuild design (spec + harness design).** ✓ DONE. `pocketmidi_rebuild_spec.md` v2, Codex design-reviewed — B3 dropped, kick-only locked, velocity guardrails added, harness gains contour-preservation metrics + multi-seed CIs + a clean train-split baseline.

**Step 4 — Build (next).** In spec order, stopping for review at each checkpoint:
1. Validation harness + baselines (quantised input, clean train-split old-schema profile = the gate, shipped-current = context).
2. Batched profile rebuild + runtime changes (velocity residual + de-bias + shrinkage, kick-only AR, relative tiering).
3. Re-measure against the acceptance gate — accept only if closer to human than both baselines on velocity fine-structure **and** contour preservation, with timing metrics unmoved.

---

## Execution playbook (which tool, which settings, when to review)

**Model:** Claude Fable 5 in Claude Code, fresh session per step (not the audit session). Mode: **ask before edits** (approve setup/test commands, decline stray edits). Thinking on. Step 4 needs GMD present locally.

**Effort per run:**
- **xhigh** — anything that runs code or is a real implementation/refactor: Step 1 (done), the Step 4 harness + rebuild.
- **high** — pure reasoning/design with no execution.
- Not max — overthinks structured work and burns usage fast. Drop to high if you hit limits.

**Review gates:**
- Step 1 code → Codex-reviewed, clean. ✓
- Step 3 spec → Codex design-reviewed before building; findings folded into v2. ✓ (the highest-value review in the plan.)
- Step 4 build → normal review after each of the three checkpoints; a fresh Claude or Codex is fine.
- **Do NOT re-review settled reviews** (the audit, or the applied Codex spec review) — that's the loop that never ends.

**Rule of thumb:** cross-model review before *irreversible* decisions (the rebuild spec ✓); single-model review after *reversible* code (the commits).