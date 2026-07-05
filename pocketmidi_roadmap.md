# pocketmidi — plan & decisions

_Last updated: 2026-07-03. Working doc: shipped state, parked backlog, and decisions. Update it as things move._

## Status at a glance
- **Step 1 — engine fixes:** ✓ shipped.
- **Step 2 — ear test:** ✓ done (confounded → led to the 0.35 default).
- **Step 3 — rebuild design:** ✓ done (`pocketmidi_rebuild_spec.md` v2 + addendum, both Codex-reviewed).
- **Step 4 — velocity rebuild (module 13):** ✓ **SHIPPED.** Live in the bundled `rock.json`, CLI serves it by default.
- **Next:** nothing required. Optional rounds parked below. Recommended move: use the tool on real tracks and let that pick the next round.

---

## What shipped (module 13 + defaults)

- **Velocity rebuild.** `vel_delta` is now the residual against its shrunk `(take, position, instrument[, tier])` mean (was: delta vs bucket median) — stops sampling accents as noise. Every emitted bucket de-biased to ~0 (verified in shipped artifact, max residual mean 3.4e-15). Snare tier-conditioned (ghost/backbeat get separate baselines). Kick-only velocity AR(1). Relative (user-percentile) velocity tiering. Shipped `rock.json` rebuilt from full GMD (341 rock takes / 114,890 hits), schema_version 2, with a regression test so an old-schema profile can't ship silently.
- **Default intensity 1.0 → 0.35.** Ear-tested: 0.3–0.35 sounds real, 1.0 sounds sloppy (it faithfully reproduces GMD's raw ~27 ms within-take spread, which is authentic but too loose as a default). Range uncapped; docs steer to 0.2–0.5 as the useful band.
- **Result:** confirmed better than the old profile by ear on both a busy ghost/backbeat pattern and a four-on-the-floor, never worse; the old profile's accent-inversion (soft hats slamming to 127) is gone.
- **Key commits:** `c6562a2` (shipped profile), `d7cf0b6` (0.35 default), `e704f09` (eartest generator + both patterns).

**Prior shipped work (Step 1):** kit-wide groove clock (fixed multi-track flams), drum-channel filter, straight non-4/4 support, phi default 0.4, docs — commits `5755df6` + `fa4a767`, Codex-reviewed.

---

## Measured facts (from the GMD pass — don't re-derive)

- Within-take timing σ 26–28 ms vs 29–31 total (between-take only ~17%). Drummers genuinely play that loose vs grid → "sloppy at 1.0" is taste, handled by the 0.35 default.
- Static per-take lean std 11 ms; true within-take wander r≈0.179 → phi≈0.21. (Why B3+phi 0.4 double-counts — see parked.)
- Velocity lag-1 autocorr: kick +0.32, snare +0.15, hats +0.07, ride +0.09 → AR is kick-only.
- Coupling tighter than humans: GMD same-slot r=0.5–0.78, σ≈15–20 ms; current ±1 ms is 10–20× tighter (taste call).
- phi calibration ~0.374 overall; ~0.35–0.45 below 130 BPM, 0.09 above 130 BPM.

---

## Open items carried forward (measured, parked — none blocking)

1. **Within-role velocity over-noise** — at full intensity, kick/hats/ride put 2–3× human spread within a fixed role (snare fixed). It's *unimodal* excess (not ghost/accent structure — bimodality is 1–3% vs snare's 10%), so tier-conditioning won't fix it; likely slow dynamic movement (crescendos/section swells) sampled as white per-hit noise. **The 0.35 default masks it in practice.** The measured next lever if you want higher intensities to hold up — would need a new correlation/conditioning mechanism (the velocity analogue of the AR timing clock). Ear verdict on it: minor.
2. **Snare zero-jump mass** marginally under the gate — human ghost-runs repeat near-identical velocities; continuous KDE residuals rarely produce exact repeats. Structural to the paradigm, not a snare bug. Cosmetic.
3. **Doc sync** — CLAUDE.md's implementation notes don't yet record module 13 / the 0.35 default. (Roadmap now updated; CLAUDE.md pending.)

### Other parked backlog (unchanged)
- **B3 (static per-song feel-offset) + phi recalibration** — paired decision; extracting the lean implies phi≈0.21, can't add B3 while phi stays 0.4 without double-counting. Revisit together, gated on a clean ear test.
- Presets (subtle/natural/loose) — if the single 0.35 default isn't enough UX.
- Tight-curated profile variant (build from the tighter half of GMD takes).
- Coupled-residual A/B (±1 ms vs ~5 ms).
- **Preserve-intent architecture** — add residual on top of user input instead of flatten-then-regenerate; fixes iterative-use (re-humanising destroys the prior pass) and the drums-in-isolation framing. Cheap first step: warn on / don't silently flatten off-grid input.
- **PyPI release-polish batch** (separate goal — "publish it", not "improve it"): pyproject license/classifiers/URLs, `pandas` → optional extra, profile rounding (~10→5 MB), CI version matrix, `--report` diagnostics.
- Known ceilings (inherent, not bugs): humanises toward the *average* GMD drummer → "convincingly generic," not distinctive; drums-in-isolation can't do mix/song awareness. Sample library is out of scope (user's kit choice).

### Revisit triggers for a learned approach (none close today)
1. Swing genres (jazz/funk) come in-scope. 2. A small embeddable open drum-humanisation model appears to *call* not reimplement. 3. Improvements built and A/B still says "programmed" (empirical ceiling). 4. "Play like drummer X from this recording."

---

## Execution playbook (for future rounds)

**Model:** Claude Fable 5 in Claude Code, fresh session per step. Mode: **ask before edits**. Thinking on. Rebuild/harness steps need GMD present.
**Effort:** xhigh for code/execution/refactor; high for pure design.
**Review gates that worked this project:** cross-review the *design/spec* with Codex before building (the expensive, hard-to-reverse fork); single-model review of *code diffs* after writing; validate *harness/measurement tools* by watching them behave, not by review; and **listen before shipping any profile as default** — the metrics ranked module 13 a pass, the ear caught that the default intensity still sounded jagged. Don't re-review settled reviews.
