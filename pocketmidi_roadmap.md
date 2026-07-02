# pocketmidi — plan & decisions

_Last updated: 2026-07-02. Working doc: locked decisions, parked backlog, and the build roadmap. Update it as things move._

---

## Locked decisions

- **Paradigm: KEEP.** Statistical delta sampling + AR(1) groove drift is the right frame and the product's reason to exist (weights-free, dependency-light, note-preserving, interpretable knobs). A learned / GrooVAE-style rewrite was evaluated and rejected — it would be a solo-maintained clone competing with Magenta Studio on its own turf. Improvements are extensions *within* this paradigm.
- **Target genres: rock + pop now.** Funk / jazz / metal are explicit *later* scope, not current weaknesses. Swing genres are the paradigm's real boundary (see revisit triggers).
- **phi default 0.5 → 0.4.** Data-backed (see facts below); pending the ear test for tempo-aware phi.
- **Module 13 pivot (velocity).** The fix is *redefining what `vel_delta` means at build time* — a residual against the `(take, grid-position)` mean instead of the bucket median. This stops sampling accent structure as noise and roughly halves applied velocity spread. AR(1) on velocity is now a **kick-only** refinement (kick autocorr +0.32; snare/hat/ride near white), plus an optional static per-song feel-offset layer. NOT "AR on all velocities."
- **Cross-instrument timing: no new mechanism.** Adjacent cross-instrument correlation ≈ same-instrument, so one shared kit clock is correct. Priority 2 reduces to the per-track-clock **bug fix** only.
- **One batched profile rebuild** against one agreed schema spec — not two (per CLAUDE.md's "batch breaking profile changes").
- **Delivery shell is open** (CLI now, maybe GUI/plugin later). Keep the core light and embeddable regardless.

---

## Measured facts (from the GMD pass — don't re-derive these)

- Within-take timing σ 26–28 ms vs 29–31 ms total → between-take is only ~17%. Drummers genuinely play that loose against the grid. "Sloppy at intensity 1.0" is **taste, not a variance bug** — fix via default/presets.
- Take tightness varies ~2× (σ 17 ms at P10 vs 37 ms at P90) → a curated "tight" profile from the tighter half of takes is cheap and viable.
- Static per-take lean: std 11 ms across takes. ~Half the pooled slot correlation (r=0.318) is this static lean; true within-take wander r=0.179 → phi ≈ 0.21. Honest model is 3 layers: per-song static feel offset (σ≈11 ms), drift clock (phi≈0.21), fast residual.
- Velocity lag-1 autocorr (within take, fixed position): kick +0.32, snare +0.15, hats +0.07, ride +0.09. Hats' scatter is **excessive magnitude, not missing correlation** — conditioning on `(take, position)` drops velocity σ from 30–39 to 15–21.
- Coupling is tighter than humans: same-slot GMD r=0.5–0.78 (kick+snare loosest 0.54, kick+crash tightest 0.78), σ≈15–20 ms. Current ±1 ms coupled residual is 10–20× tighter — a defensible taste call for produced music, worth an A/B (~5 ms) later.
- phi calibration: recommended ~0.374 overall; tempo split ~0.35–0.45 below 130 BPM, **0.09 above 130 BPM** (fast rock is nearly hit-to-hit independent).

---

## Bugs & fixes queued (from the audit)

- **CRITICAL — per-track groove clock flams multi-track files** (up to ~73 ms on separate kick/snare tracks). Fix: step the clock and detect chords on one absolute-time-ordered stream merged across tracks; keep windowing/delta reconstruction per-track.
- **HIGH — no drum-channel filter**; melodic parts on drum-range notes get corrupted. Default to channel 10, `--all-channels` opt-out.
- **MEDIUM — 3/4 rejected despite docs.** Accept straight non-4/4 meters via the `grid_pos=None` path (6/8 precedent).
- **MEDIUM — velocity de-bias.** Subsumed by the module-13 pivot (within-`(take,pos)` residual baseline).
- **LOW / product** — intensity default (→ 0.4 or presets), `--report` diagnostics, `pandas` → optional extra, profile rounding (~10 MB → ~5 MB), pyproject license/classifiers/URLs, CI version matrix, `AGENTS.md` sync.

---

## Parked backlog (write-down, don't touch now)

- **Validation harness** — promoted to a build step (see roadmap). Listed here as the anchor for everything stochastic.
- Ear test → tempo-aware phi (the ear test itself is a *now* action; tempo-aware phi is parked pending it).
- Tight-curated profile variant.
- Coupled-residual A/B (±1 ms vs ~5 ms).
- **Preserve-intent architecture** — add human residual *on top of* the user's input instead of flatten-to-grid-then-regenerate. Answers both the drums-in-isolation and iterative-use problems. Cheap first step: warn on / don't silently flatten off-grid input.
- Feed-the-song / groove-transfer ("play like this recording") — out of scope; it's revisit-trigger territory, not a knob.
- Known ceiling: humanising toward the *average* GMD drummer kills robotic but caps at "convincingly generic," not "distinctive feel." Fine for rock/pop backing.

### Revisit triggers for a learned approach (none close today; watch #3)
1. Swing genres (jazz/funk) come in-scope — per-slot stats can't represent phrase-dependent swing.
2. A small, permissive, embeddable open drum-humanisation model (GrooVAE-class, <50 MB, ONNX) appears to *call* rather than reimplement.
3. **Priorities 1–3 are built and A/B listening still says "programmed."** ← the empirical ceiling test.
4. Users want "play like drummer X from this recording" (an embedding feature).

---

## Roadmap (in order)

**Step 1 — Safe engine fixes (no GMD needed).**
Commits: drum-channel filter · accept straight non-4/4 · README/AGENTS truth-up · kit-wide groove clock · phi default 0.4. Ship-ready today.

**Step 2 — Ear test.**
`demo/eartest/fast_145bpm_phi01.mid` vs `phi05.mid` in Logic. 5 minutes. Only thing gating tempo-aware phi.

**Step 3 — Design the rebuild + build the validation harness.**
Write (a) the module-13 / profile schema spec and (b) a validation harness: hold out GMD tracks, quantise them, re-humanise, measure how close the *output statistics* land to the real human performance (offset/velocity distributions, lag-1 autocorrelations, cross-instrument gaps). Success = re-humanised sits measurably closer to human than the quantised version. This is the ground-truth "did it get more human, not just different" gate for every layer.

**Step 4 — One batched rebuild.**
Rebuild profiles against the agreed schema; implement module 13 (within-`(take,pos)` residual baseline + kick AR + optional static feel layer). Judge with the harness **and** ears, not ears alone.

---

## Execution playbook (which tool, which settings, when to review)

**Model:** Claude Fable 5 for all of it, in Claude Code. Mode: **ask before edits** for every run (approve setup/test commands, decline stray file edits). Thinking on.

**Effort per run:**
- **xhigh** — anything that runs code or is a real implementation/refactor: Step 1 (esp. the kit-clock refactor), building the harness, the Step 4 rebuild.
- **high** — pure reasoning/design with no execution: writing the schema spec.
- Not max — overthinks structured work and burns usage fast. Drop to high if you hit limits.

**Review gates:**
- Steps 1 & 4 code → normal review after writing. A fresh Claude agent or Codex is fine; these are mostly unambiguous.
- **Step 3 schema spec → cross-review with Codex *before* building.** This is the one expensive, hard-to-reverse fork; a different model catches correlated blind spots. This is the highest-value review in the whole plan.
- **Do NOT re-review the audit** — it's verified measurement, not opinion. Re-auditing just invites bikeshedding.

**Rule of thumb:** cross-model review before *irreversible* decisions (the rebuild spec); single-model review after *reversible* code (the commits).
