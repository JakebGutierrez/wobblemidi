# wobblemidi — porting contract & handoff

_Frozen 2026-07-06 (Fable window, Session 3); amended 2026-07-07 (bonus session): the
Tier 2(b) runner is now built and calibrated — section 2(b) below is a concrete locked
procedure, no longer a future deliverable. Audience: future sessions (human or model)
that must inherit conclusions instead of re-deriving them, and whoever executes the
JUCE port. Everything in "Settled decisions" is a fact, not a proposal — do not reopen
these without the owner explicitly doing so._

## Settled decisions (facts)

1. **v1 (this repo) ships first** — as a portfolio piece and, more importantly, as the
   **reference implementation** the port is validated against.
2. **The endgame is a Logic plugin, non-negotiable.** Target: **JUCE / C++ / AU** (AU
   MIDI-processor type). A standalone app is a JUCE build target of the same codebase,
   not a separate product.
3. **Interaction model: Superior Drummer 3-style drag-in / process / drag-out.** The
   plugin contains an **offline MIDI environment operating on whole clips**. It is NOT a
   real-time streaming MIDI FX. Whole-clip context (relative velocity tiering, coupling,
   future section mode / bass-follow) is preserved by design —
   `wobblemidi_streamability.md` is the inventory proving this model needs no redesign.
4. **No rewrite of the Python app.** Python is not a bottleneck for the offline engine;
   this codebase's job is to be the reference the JUCE core is checked against.
5. **The harness is the source of truth, not any prose spec.** A port is correct when it
   passes the harness and matches the reference — not when it matches a document
   (including this one: on conflict, code + vectors + harness win).
6. **FOSS, stays FOSS.**
7. **Working discipline:** engine changes gate on design review, then independent code
   review, before merge. Docs/scripts/tests go direct. Listen before shipping any
   profile or default change (the metrics passed module 13; the ear caught the
   intensity default).

## Port correctness — the two-tier definition

**Tier 2 is the gate. Tier 2 alone means the port is correct.**

### Tier 2 (sufficient on its own)

(a) **Harness pass:** `scripts/validate.py` on held-out GMD, within the recorded
envelope (below). The harness measures distance-to-human per instrument on programmed
inputs at `intensity=1.0` (full-scale engine-vs-human comparison, independent of the
0.35 product default), two-sided against human — never "beat the input".

(b) **Distributional equivalence against the reference engine** — built, calibrated
and locked 2026-07-07. The concrete procedure:

- **Runner:** `python scripts/compare_port.py verify <candidate_dir>`. Cells = the
  golden vectors (deduplicated: `f1_cli`, `f1_seed7` skipped) + the runner-owned
  full-kit coverage fixture `tests/tier2/t2_full_kit.mid` (all 22 TD-11 notes — the
  golden inputs only cover 36/38/42/46/49). The port supplies **exactly 32
  independent-seed outputs per distributional cell** (layout `<dir>/<cell_id>/*.mid`,
  ≥1 file per identity cell), seeds free to differ; the runner regenerates its own
  pinned-seed reference pools, self-checks the local engine against the golden
  manifest first, and exits 0/1. `python scripts/compare_port.py dump-reference <dir>`
  writes example pools + the expected layout for port developers;
  `--cells a,b` supports partial verification while porting module by module.
- **Gates:** `calibration/tier2_thresholds.json` — 799 comparisons (harness-suite
  metrics per instrument and per cell, plus per-note gates on the full-kit cell and
  0.5 ms absolute gates on deterministic behaviours: rigid-cluster gaps, note_off
  placement, chord tightness) + 13 aggregate family z-gates + ungated structural
  contracts (alignment, melodic passthrough, meta preservation, mode-flag
  invariants, identity cells). Metric functions come from `scripts/validate.py`, the
  same scoring core the harness uses.
- **Calibration (empirical, not fiat):** thresholds = null q99.5 × margin 1.62,
  from 400 reference-vs-itself replicates; margin = the observed null max-ratio,
  validated by **20/20 held-out full-verdict nulls**. A 13-mutant battery
  (timing/velocity scale, phi and PHI_VEL, tiering, KDE bandwidth, coupling ×2,
  de-bias, mapping ×2) detects 12 at 1.15–12.7× the bands. Evidence, stability
  analysis and reproduction: `calibration/tier2_calibration.md` (committed,
  self-sufficient in-tree). Bands are valid **only at K=32 pools**; any change goes
  through `scripts/calibrate_tier2.py` + a fresh sign-off.
- **Recorded limitation:** a PHI_VEL ±0.1 transcription error sits below the K=32
  noise floor (the not-ported failure mode IS detected at 2.1×, and the Tier 2(a)
  envelope pins kick velocity lag-1 vs human). Escalation: rerun both pools at
  higher K (e.g. 128) on `t2_full_kit`/`f3_default` — estimator noise shrinks ~√K
  while a real coefficient error does not; informational, since the locked bands
  hold only at K=32.
- **False-failure semantics:** a correct port failing marginally (ratio ≈ 1.0) may
  regenerate its pools with fresh seeds — seeds are free by contract; every
  calibrated defect shows ≥1.15× and most 2–13×.

### Tier 1 (optional, gold standard)

Byte-match all golden vectors: `scripts/verify_golden.py` semantics against the port's
output. Only achievable by reimplementing the exact RNG/KDE machinery — the algorithm
inventory in `wobblemidi_determinism.md` (numpy legacy `RandomState`, scipy
`gaussian_kde.resample`, the seed derivations, half-to-even rounding). Worth doing if
the port embeds a MT19937 + the KDE resample math; not required for correctness.

**RNG recommendation (2026-07-07):** the C++ core should use its **own clean RNG**
(e.g. `std::mt19937_64` or a PCG, with straightforward normal/uniform draws) rather
than attempting MT19937/scipy mimicry — Tier 2 is the gate and it is seed-free by
design; Tier 1 byte-match remains the optional gold standard for anyone who later
chooses to chase it.

## The verification artifacts (what exists, how to run it)

| Artifact | Role | Run |
|---|---|---|
| `tests/golden/` (10 inputs, 26 vectors, manifest) | Byte-level contract of current engine behaviour; regression lock for ALL future Python changes; Tier 1 target and Tier 2(b) input set | `python scripts/verify_golden.py` (also in every pytest/CI run via `tests/test_golden_vectors.py`) |
| `scripts/make_golden.py` | Regenerator. **Regeneration = redefining the contract**: only alongside an intentional behaviour change, same commit, `--force` required | — |
| `wobblemidi_determinism.md` | Seed semantics, three RNG streams, draw order, rounding rules, guarantees + hazards | — |
| `wobblemidi_streamability.md` | Plugin-readiness map (offline-clip model confirmed) | — |
| `scripts/validate.py` | Part C harness: distance-to-human on held-out GMD; gates profile rebuilds and grades ports (Tier 2a) | `python scripts/validate.py <path/to/groove-v1.0.0>` |
| `scripts/compare_port.py` + `tests/tier2/t2_full_kit.mid` | Tier 2(b) runner: port-vs-reference distributional equivalence on the golden inputs + full-kit fixture | `python scripts/compare_port.py verify <candidate_dir>` (self-test: `… self-null`) |
| `calibration/tier2_thresholds.json` + `calibration/tier2_calibration.md` | Locked Tier 2(b) bands (margin 1.62, K=32) + the null/mutation evidence behind them | regenerate via `scripts/calibrate_tier2.py` + re-sign-off only |
| Full test suite (325) | Engine invariants + golden locks + Tier 2 runner locks (self-null PASS / gross mutant FAIL) | `.venv/bin/pytest` |

**Harness setup:** download the Groove MIDI Dataset v1.0.0 (Magenta, Roland TD-11) and
pass its directory (containing `info.csv`) to `validate.py`. The **train/test split
comes from GMD's own `split` column** (rock = `style.str.startswith("rock")`; gate
profile builds from `train`, evaluation runs on held-out `test`; `validation` split
unused) — profiles are never built from data the harness tests against. Reports land in
`validation/` (gitignored — local artifacts; the durable record is below and in
`wobblemidi_roadmap.md`).

## Recorded harness envelope (the numbers a port is graded against)

Measured on GMD rock (341 takes / 114,890 hits; full history in
`wobblemidi_roadmap.md` "Measured facts" and CLAUDE.md module 13 — don't re-derive):

- **Timing:** within-take σ 26–28 ms vs 29–31 ms total (between-take only ~17%) — the
  engine at `intensity=1.0` must reproduce this scale, and does; 0.35 default ≈ σ 10 ms
  on a 95 BPM pattern is the ear-chosen product point.
- **Timing correlation:** phi calibration ≈ 0.374 overall (0.35–0.45 below 130 BPM,
  0.09 above); shipped default `phi=0.4`. Static per-take lean std 11 ms; true
  within-take wander r≈0.179 (the parked B3+phi≈0.21 pairing — do not add B3 without
  recalibrating phi).
- **Velocity:** lag-1 autocorr kick +0.32 / snare +0.15 / hats +0.07 / ride +0.09 →
  kick-only AR (`PHI_VEL=0.37`). Post-rebuild within-cell residual σ: snare 9.9
  (tier-conditioned), kick 14.1, hats 20.8.
- **Coupling:** GMD same-slot cross-instrument r=0.5–0.78, σ≈15–20 ms; the engine's
  ±1 ms coupled residual is 10–20× tighter than human — a deliberate taste call, locked
  in the vectors.
- **Known misses, accepted and recorded (a port must not silently "fix" these — match
  the reference, then improve deliberately):** (1) within-role velocity over-noise at
  full intensity — kick/hats/ride carry 2–3× human within-role spread (human ~6.4–8.5);
  unimodal excess, masked at the 0.35 default; the measured next lever is a velocity
  analogue of the AR timing clock. (2) Snare zero-jump mass marginally under the gate —
  continuous KDE residuals rarely emit the exact velocity repeats of human ghost runs;
  structural, cosmetic.

## Velocity-rebuild verdict (closes the window's conditional item)

**Ship-as-is.** The harness evidence that would have triggered a "velocity rebuild
design note" instead triggered the full rebuild during this window's earlier sessions:
module 13 shipped (`c6562a2`) — residual `vel_delta` schema v2, snare tier-conditioning,
kick velocity AR, relative tiering — harness-gated on held-out GMD and ear-confirmed
better-never-worse on both test patterns. The two recorded misses above are parked with
rationale (masked at the 0.35 default; next lever identified). No further velocity work
is required before the port.

## Reference-implementation inventory (what a port implements)

- **Engine:** `wobblemidi/humanise.py` (three-pass architecture: per-track precompute →
  one merged kit-wide pass → per-track delta re-encode), `wobblemidi/midi_utils.py`
  (TD-11 mapping, tempo map, grid). CLI: `wobblemidi/cli.py`. The GUI seam
  (`wobblemidi_gui/`) is NOT part of the engine contract.
- **Profile:** `wobblemidi/profiles/rock.json`, schema v2. Bucket keys
  `genre|beat_type|instrument[|tier][|grid_pos]` plus pooled `global|instrument`;
  values `[[offset_ms, vel_delta], …]`; `_meta`: `schema_version`,
  `vel_delta_definition`, `velocity_thresholds`, `kde_bw_method`,
  `bucket_offset_means`, `bucket_vel_delta_means`, `vel_sigma_within`,
  `tier_residual_groups`. `vel_delta` is a residual vs the shrunk
  `(take, position, instrument[, tier])` mean — see CLAUDE.md module 13 before touching.
- **Parameter surface** (pinned per golden vector in `tests/golden/manifest.json`):
  `genre, beat_type, intensity (default 0.35), seed, timing_only, velocity_only, push,
  phi (default 0.4), all_channels, push_amount, intensity_by_group`. CLI exposes all but
  `push_amount` and `intensity_by_group` (API-only, GUI-facing; CLI flags deferred).
  Validation rules at `humanise.py:416-449`.
- **Behavioural spine** (each locked by vectors and/or tests): 16th grid, 8th for 6/8,
  positional buckets only on 4/4; six-level fallback chain; per-file relative tiering;
  de-bias vs `--push`/lean mirror; kit-wide AR(1) timing clock + kick velocity clock;
  same-tick coupling + 12 ms windowed rigid clusters with cluster-scope clamping and
  elastic member offs; drum-channel filter; type 0/1 only; 6/8-mix rejection.

## For future sessions (how to not break this)

- Any Python engine change: full pytest must stay green — the golden locks make "did
  behaviour change?" a yes/no question. An intentional behaviour change regenerates
  vectors (`make_golden.py --force`) **in the same commit**, with the review gate from
  settled decision 7.
- Any profile rebuild: gate through `scripts/validate.py` (train-built gate profile,
  held-out test), then **listen** (`scripts/make_eartest.py`), then regenerate vectors.
- The port: implement from this inventory, grade with Tier 2 — the 2(b) runner is
  already built, calibrated and pytest-locked (port a module, produce 32-seed pools,
  run `compare_port.py verify`, red or green) — optionally chase Tier 1 with its own
  clean RNG in the meantime (see the RNG recommendation above). JUCE work was
  explicitly out of scope for this window; nothing here presupposes any scaffolding
  choice beyond decisions 2–3.
