# wobblemidi — Claude Code context

## What this is
CLI tool that humanises programmed drum MIDI using real drummer timing/velocity
distributions from the Groove MIDI Dataset (Google Magenta, Roland TD-11).

## Dev setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Venv lives at `.venv/` — always use `.venv/bin/pytest` etc.

## Running tests
```bash
.venv/bin/pytest               # all tests
.venv/bin/pytest tests/test_midi_utils.py -v   # specific module
```

All tests must pass before moving to the next module.

## Build order (current status)
| # | Module | Status |
|---|--------|--------|
| 1 | Scaffold (pyproject.toml, dirs) | done |
| 2 | `wobblemidi/midi_utils.py` | done |
| 3 | `scripts/build_profiles.py` | done |
| 4 | `wobblemidi/humanise.py` | done |
| 5 | `wobblemidi/cli.py` | done |
| 6 | `tests/test_humanise.py` | done |
| 7 | `--timing-only` / `--velocity-only` flags | done |
| 8 | Velocity-stratified buckets + KDE sampling | done |
| 9 | Grid position awareness | done |
| 10 | Outlier clipping | done |
| 11 | 6/8 support | done |
| 12 | Groove drift + coupled hits (`--groove-tightness`) | done |
| 13 | Velocity rebuild (schema v2: residual vel_delta, kick VelDrift, relative tiering) + validation harness | done |

Build one module at a time. Use plan mode for each new module.

## Workflow
- Plan mode before any multi-file or new-module work
- Read existing code before editing
- Commit after each working module
- Never add Co-Authored-By lines to commit messages

## Design decisions — do not change without discussion

**Instrument mapping:** Roland TD-11 only. Notes 22 and 26 are hi-hat edge
variants not in the GM spec — they must stay in `TD11_TO_GROUP`. See
`wobblemidi/midi_utils.py`.

**Grid:** 16th-note grid by default; 8th-note grid for 6/8 files. No swing/triplet in v1.

**Time signature:** Auto-detected via `detect_meter()` in `midi_utils.py`. 6/8 files
use an eighth-note grid (`ppq // 2`) and skip grid-position bucket lookups (pass
`grid_pos=None` — positional buckets assume a 4/4 bar). Files that mix 6/8 with any
other signature are rejected (a single grid choice cannot represent both sections).
3/4 and other uniform quarter-note-based meters work on the 16th grid as-is.

**MIDI file type:** Type 0 and type 1 only. `humanise.py` builds a single song-level
tempo map and applies it across all tracks. Type 2 (independent per-track timing) is
not supported — GMD files are type 0/1.

**Genre filter:** Rock only in v1. Filter: `df[df["style"].str.startswith("rock")]`.
Jazz/funk profiles are explicitly out of scope (swing feel is misrepresented under
a straight 16th grid).

**Bucket key:** `(genre, beat_type, instrument_group)` — no grid position until v2.

**Profile storage format:** List of `(offset_ms, vel_delta)` tuple pairs.
Do NOT store as separate lists — that breaks the v2 KDE upgrade path.
`offset_ms`: positive = late, negative = early.
`vel_delta` (schema v2, module 13): velocity residual vs the shrunk
`(take, grid_position, instrument[, tier])` mean, each emitted bucket de-biased to
mean ~0 — NOT raw velocity, and no longer the v1 bucket-median delta (that sampled
the user's accent structure as noise). See module 13 notes.

**Sparse fallback order:**
1. `(genre, beat_type, instrument)` — exact
2. `(genre, "beat", instrument)` — drop fill context
3. `("global", instrument)` — pooled
4. no change applied

**Ghost note filter:** `VELOCITY_FLOOR = 20` for kick and snare only during
profile build. Hi-hats and cymbals are exempt.

**Intensity:** Scales sampled deltas linearly toward zero —
`applied = sampled * intensity`. Do not clamp before scaling. Default **0.35**
(ear-tested, 2026-07); 1.0 reproduces GMD's raw within-take spread and sounds
loose — see the default-intensity notes below. Range stays un-capped.

**v2 upgrade point:** `humanise.py` samples offsets and vel_deltas independently
in v1. The v2 upgrade replaces this with `scipy.stats.gaussian_kde` — the tuple
pair storage format is designed to make this a drop-in replacement.

## v2 priorities (from real-world testing)

Ordered by musical impact:

1. **Velocity-stratified buckets** — v1 treats all snare hits identically regardless
   of velocity. Ghost notes (low velocity) and backbeats (high velocity) need separate
   buckets: `rock|beat|snare|soft`, `rock|beat|snare|medium`, `rock|beat|snare|hard`.
   Thresholds should be derived from actual GMD velocity distributions, not guessed.
   Fallback chain gains one extra level: exact tier → drop tier → drop fill → global.
   GMD rock|beat|snare has ~26,825 samples — enough to stratify even if soft hits are
   10% of that.

2. **`--timing-only` / `--velocity-only` flags** — done. See implementation notes below.

3. **KDE sampling** — replace flat independent sampling with
   `scipy.stats.gaussian_kde`. Storage format already supports this as a drop-in.
   Do this alongside velocity stratification (both require rebuilding profiles).

4. **Grid position awareness** — bucket key becomes
   `(genre, beat_type, instrument, grid_position)`. Beat 1 kick vs off-beat kick
   have different timing tendencies in real drumming.

5. **Custom profile source** *(deferred — out of scope for now)* — `--profile
   path/to/custom.json` flag so users can build profiles from their own MIDI packs
   (e.g. professional drummer sample packs) and humanise to sound like a specific
   player. Revisit if there is a concrete use case requiring a non-GMD source.

6. **6/8 support** — done. See implementation notes below.

**Do items 1 + 3 together** — both require rebuilding profiles and changing the
bucket key structure. Breaking change, worth batching.

## Implementation notes — completed modules

### build_profiles.py
Run: `python scripts/build_profiles.py <path/to/groove-v1.0.0>`
Output: `wobblemidi/profiles/rock.json`

Non-obvious implementation decisions:
- **Offset computation:** `offset_ticks_to_ms` (scalar tempo) is NOT used. Instead,
  `ticks_to_ms_with_map` is called for both legs of the offset delta so that any tempo
  change falling between `grid_tick` and `abs_tick` is handled correctly.
- **Tick accumulation:** `abs_tick` resets to 0 per track, not per channel — MIDI delta
  times are track-local.
- **Global buckets:** In addition to `rock|{beat_type}|{instrument_group}` keys, the
  script also writes `global|{instrument_group}` keys (all beat_types pooled) to support
  fallback level 3. Median velocity is computed independently per bucket.
- **JSON key format:** `"genre|beat_type|instrument_group"` for exact buckets;
  `"global|instrument_group"` for pooled. Values are `[[offset_ms, vel_delta], ...]`.
- **File/parse errors** are silently skipped with a counter; the script continues.

## Implementation notes — module 8: velocity-stratified buckets + KDE sampling

Requires rebuilding `wobblemidi/profiles/rock.json`. Batch items 1 + 3 together —
both require a profile rebuild and changes to the bucket key structure.

**Velocity tier thresholds — use tertiles, not fixed values.** Split soft/medium/hard
at the 33rd and 66th percentile of actual GMD velocities per instrument group. This
lets the data decide where the boundaries are rather than guessing. Compute per
instrument (kick and snare have different typical velocity ranges). Write computed
thresholds to a `_meta` key in the JSON so `humanise.py` reads them at load time.

**KDE — fit at load time, not per hit.** Fitting a KDE is expensive; sampling from
one is cheap. Fit once in `load_profile` when the JSON is read, store the fitted KDE
objects in the profile dict. Never refit inside the per-hit loop.

**KDE bandwidth — check by ear after first rebuild.** scipy's default (Scott's rule)
is the right starting point. However, hi-hat timing in GMD can have two clusters —
right on the grid and slightly behind — and Scott's rule may blur these into one,
making the hi-hat feel smeared. After the first profile rebuild, listen to a hi-hat
pattern and check that it has a sense of pocket rather than random scatter. If it
sounds wrong, try Silverman's rule or a manually set bandwidth in `build_profiles.py`.
This is a developer tuning step, not a user-facing flag — bandwidth has no meaningful
musical label and should not be exposed as a CLI option.

## Implementation notes — --timing-only / --velocity-only

- **`velocity_only` bypasses the timing path entirely** via `continue` after appending
  `(abs_t, msg.copy(velocity=new_vel))`. Do NOT seed `candidate = abs_t` and let the
  windowing run — the `prev_note_on_abs` lower bound would still bump same-tick notes
  by `EPSILON_TICKS`, breaking the "position untouched" contract.
- **Random draws** (`i`, `j`) always happen regardless of mode so RNG state stays
  consistent when toggling flags with the same seed.
- **Mutual exclusion** is enforced in both `humanise()` (raises `ValueError`) and
  `cli.py` (exits 1 before profile load). CLI guard runs first so no file I/O happens.
- **Profile validation** is intentionally deferred: `load_profile` assumes well-formed
  `[[offset_ms, vel_delta], ...]` pairs. Add shape/type validation when `--profile`
  (user-supplied JSON) is implemented.

## Implementation notes — cli.py

### cli.py
Entry point: `wobblemidi <input.mid> <output.mid>`

Non-obvious implementation decisions:
- **Profile resolution:** Genre maps to `wobblemidi/profiles/{genre}.json` via
  `importlib.resources.files("wobblemidi.profiles").joinpath(...)` + `as_file()`.
  `as_file()` is required (not `str()`) to guarantee a real filesystem path in all
  install layouts (editable, wheel, zip-imported).
- **`--section` flag:** User-facing name for `beat_type` — maps directly to the
  `beat_type` parameter of `humanise()`.
- **`--intensity` validation:** Uses `click.FloatRange(0.0, 1.0)` — Click rejects
  out-of-range values before `humanise()` is called.
- **Packaging:** `[tool.hatch.build] include` covers both wheel and sdist so
  `wobblemidi/profiles/*.json` ships in all distribution formats.

## Implementation notes — module 9: grid position awareness

Bucket key gains a 16th-note grid-position dimension (0–15 within a 4/4 bar).
`grid_position_in_bar(grid_tick, ppq)` uses `16 * (ppq // 4)` for bar length —
NOT `ppq * 4` — so the wrap is consistent with the truncated `quantise_to_grid`
sixteenth. Using `ppq * 4` breaks for any PPQ not divisible by 4.

Stratified fallback chain with grid_pos (6 levels):
1. `rock|beat|kick|hard|3` — tier + grid_pos
2. `rock|beat|kick|3` — unstratified + grid_pos (keeps position signal past tier miss)
3. `rock|beat|kick|hard` — tier only
4. `rock|beat|kick` — style only
5. `rock|beat|kick` — drop fill context
6. `global|kick`

`grid_pos=None` → offset=0 → levels 1,2,3,4 unchanged (backward compat).

Grid-position lookups only run on 4/4 files (positional buckets assume a 4/4
bar). Non-4/4 files are accepted and pass `grid_pos=None`, falling back to the
per-instrument ms deviation buckets — the 4/4 rejection gate was removed in the
Step 1 engine fixes (see below).

rock.json rebuilt from GMD: 315 buckets (277 grid-position, 38 legacy fallback),
7 non-4/4 files skipped.

## Implementation notes — module 10: outlier clipping

Clips `offset_ms` to the 2nd–98th percentile **per bucket** in `build_profiles.py`
before KDE fitting, removing accidental timing errors (drummer mistakes) from GMD.

Key design decisions:
- Clipping logic lives in `_clip_hits()` — a helper called by both
  `_build_pairs_with_clip()` (public contract, tested directly) and
  `_build_profiles()` (needs both pairs and mean from the same retained set).
- `_build_pairs_with_clip()` itself is a thin wrapper so its `list | None` return
  contract and existing unit tests are unchanged.
- MIN_SAMPLES gate is enforced on the **retained** (post-clip) set. Buckets that
  shrink below 30 are skipped and fall through to the correct fallback.
- `_build_pairs` itself is unchanged — it receives the already-clipped hit list.

rock.json rebuilt: 311 buckets (4 fewer than pre-clip — near-threshold buckets
that shrank below MIN_SAMPLES after clipping).

## Implementation notes — module 11: 6/8 support

**`detect_meter(midi_file) -> str`** in `midi_utils.py`:
- Collects all `time_signature` events with absolute tick positions across all tracks.
- Returns `"6/8"` only if every event is 6/8 AND the first is at tick 0 (no implicit
  4/4 prefix).
- Returns `"non-6/8"` when no 6/8 events are present — includes no events (MIDI default
  4/4), uniform 4/4, uniform 3/4, and non-6/8 mixed-meter files (16th grid is valid for
  all quarter-note-based meters, so non-6/8 mixing does not require rejection).
- Raises `ValueError` if 6/8 is mixed with any other signature, or if the first 6/8
  event is not at tick 0 (implicit 4/4 region before it counts as mixing).

**`quantise_to_grid(time_ticks, ppq, grid="16")`** — `"8"` uses `ppq // 2` subdivision.
Default `"16"` is unchanged; all existing callers are unaffected.

**`humanise()` changes:**
- Calls `detect_meter(mid)` immediately after the type-2 check; sets `grid = "8"` for
  6/8, `"16"` otherwise.
- Both note-processing loops pass `grid` to `quantise_to_grid`; positional lookup
  runs only when `use_grid_pos` (4/4 files) — 6/8 and other non-4/4 meters pass
  `grid_pos=None`, which skips the positional bucket chain entirely and falls back
  to the per-instrument ms deviation buckets, which transfer well across meters.

**No profile rebuild needed.** The existing rock.json is reused unchanged.

## Implementation notes — --push flag / offset de-bias

**Problem:** GMD rock drummers genuinely push kick (and other instruments) ahead of
the beat at certain grid positions. At low intensity (e.g. 0.3) this creates a
systematic early lean that sounds like a drummer rushing, not intentional feel.

**Design:** Separate *systematic tendency* (push/pull) from *human variation* (spread).
Per-bucket mean offsets are stored in `_meta.bucket_offset_means` at build time.
At sample time in `humanise()`, de-bias is applied by default — the bucket mean is
subtracted from `offset_ms_raw` before intensity scaling.

`--push` restores the original GMD behaviour for users who want the authentic lean.

**For README / end users:**
```
--push      Include the directional timing tendencies of the GMD source
            drummers. Without this flag (default), timing variation is
            centred on the grid — natural human imprecision without
            systematic push or drag. Use --push if you want a specific
            "leaning into the beat" feel that matches the original recordings.
```

Key implementation decisions:
- `_clip_hits()` extracts the retain/clip logic so `_build_profiles()` can compute
  `mean = np.mean([h["offset_ms"] for h in retained])` from the same retained set
  used for KDE fitting, without changing `_build_pairs_with_clip()`'s return contract.
- `_build_profiles()` returns `(profiles, bucket_offset_means, written, skipped)`.
- `LoadedProfile` gains `bucket_offset_means: dict[str, float]` with
  `default_factory=dict` — all existing `LoadedProfile(...)` call sites are unaffected.
- `_lookup()` return type extended to `(BucketProfile | None, int | None, str | None)`.
  The matched key is used to look up the mean; level and its meaning are unchanged.
- De-bias is applied to `offset_ms_raw` from `_sample_bucket()` before intensity:
  `offset_ms = offset_ms_raw - profiles.bucket_offset_means.get(key_used, 0.0)`.
  Old profiles without `bucket_offset_means` in `_meta` → empty dict → 0.0 correction
  (backward compatible, behaviour is push=True equivalent on old profiles).
- `_build_pairs` and `_sample_bucket` are unchanged.
- Requires profile rebuild to populate `bucket_offset_means` in `_meta`.

## Implementation notes — module 12: groove drift + coupled hits

Replaces independent per-hit timing with **one AR(1) drift clock** plus **coupled
(same-tick) hits**. Kit-wide across all tracks since the Step 1 engine fixes (originally
per-track — see below). User-facing knob: `--groove-tightness` (phi, default 0.4). No
profile rebuild — reuses the existing `rock.json`.

**`GrooveDrift` class in `humanise.py`.** A shifted *solo* hit does
`drift = phi*drift + sqrt(1-phi**2)*c` (AR(1) on the mean-centred sample `c`), then
`fluct = sqrt(1-RESIDUAL_SHARE)*drift + sqrt(RESIDUAL_SHARE)*residual`. This is a genuine
variance split (`Var(fluct) == Var(c)` for a stationary bucket), not the AR innovation
relabelled — the task asked for "drift **plus** a small independent residual". Constants
`RESIDUAL_SHARE = 0.15`, `COUPLED_RESIDUAL_FRAC = 0.15`, `COUPLED_RESIDUAL_MS = 1.0` are fixed
internal tunings; only phi is exposed. Calibrating phi from GMD is deferred.

Key design decisions (all from three plan reviews — do not undo without re-checking):
- **`phi == 0.0` is an exact bypass** in `humanise()`: `offset_ms = offset_ms_raw - (0.0 if
  push else mu_debias)` — byte-identical to the pre-module-12 path (no-push) / output-identical
  (push), with drift **and** coupling inert. This is the "phi=0 reproduces today" contract and
  is what makes coupling phi-gated. Keep this branch literally equal to the old arithmetic.
- **Clock is centred on the bucket's own sample mean** (`mu_center = bucket.offsets.mean()`),
  not on `bucket_offset_means`. This zero-means `c` even for legacy/meanless profiles, so their
  systematic lean is **never amplified** by `sqrt(1-phi**2)/(1-phi)`; the lean is re-applied
  statically via `mu_center - (0.0 if push else mu_debias)`. Preserves the documented
  "old profile = push-equivalent" behaviour with no profile/doc change.
- **Residual RNG is separate** (`resid_rng`, a `RandomState` seeded reproducibly from `seed`,
  NOT the global `np.random`). So the offset/velocity **sample** stream is identical across phi
  — a phi A/B is a clean timing-only comparison, and phi=0 vs phi>0 draw the same samples.
  Do not route the residual through the global stream.
- **Coupled hits (`abs_t == chord_tick`)** land at the anchor solo hit's **actual emitted tick**
  (`chord_anchor_abs`, captured *after* window-clamping) plus a `±COUPLED_RESIDUAL_MS`-capped
  residual, and do **not** advance the clock. Targeting the real landing (not the anchor's
  pre-clamp desired offset) keeps the chord tight even when the anchor is window-clamped or a
  fixed same-tick event sits between members — otherwise a coupled member could fly ~40 ms late
  (regression: `test_coupled_stays_tight_with_interspersed_fixed_note`). The static lean is
  inside the anchor's landing, so coupling stays tight under `--push` too. `chord_tick` is set
  only by a real shiftable solo hit (guards against false coupling). Separate from the patch's
  `prev_note_on_orig_abs` windowing chord logic, which stays and applies at all phi.
- `velocity_only` `continue`s before the timing block → clock never advances.

**Output autocorrelation is `(1 - RESIDUAL_SHARE) * phi`**, not phi (the independent residual
dilutes the drift's phi-autocorrelation). The exact-constant tests that asserted this value
were retired in module 13 (O5) in favour of invariants — do not reintroduce exact-constant
locks; see module 13 notes.

**Demo:** `scripts/make_demo.py` writes `demo/rock_4bar_{input,phi0,phi05}.mid` — same seed,
phi 0.0 vs 0.5, for A/B listening in a DAW.

## Implementation notes — Step 1 engine fixes (2026-07)

Safe engine fixes from the roadmap (`wobblemidi_roadmap.md`) — no GMD, no profile rebuild.

- **Drum-channel filter:** `will_shift` requires `msg.channel == DRUM_CHANNEL` (9,
  i.e. MIDI channel 10) unless `all_channels=True` (`--all-channels` in cli.py).
  Melodic parts on drum-range note numbers pass through untouched and act as fixed
  events (windowing bounds), same as any other unshifted note.
- **Straight non-4/4 meters accepted:** the 4/4 rejection gate is gone. A single
  `use_grid_pos = meter != "6/8" and is_four_four(mid)` decides positional lookups;
  non-4/4, non-6/8 files (3/4, 5/4, non-6/8 mixes) get `grid_pos=None` and use the
  per-instrument buckets — the 6/8 precedent. 6/8-mixed files still raise in
  `detect_meter()`.
- **phi default 0.4** (was 0.5) in both `humanise()` and `--groove-tightness`.
  Data-backed: GMD calibration recommends ~0.374 (`scripts/calibrate_phi.py`).
- **Kit-wide groove clock:** `GrooveDrift`, `chord_tick`, and `chord_anchor_abs` moved
  out of the per-track loop. `humanise()` now runs three passes: (1) per-track
  precomputation (`abs_messages`, `will_shift`, `next_fixed`, `paired_note_off_abs` —
  logic unchanged), (2) ONE global loop over all messages merged across tracks and
  stable-sorted by absolute tick (ties keep track order, then message order), with
  kit-wide clock/chord state and per-track windowing state indexed by track, (3)
  per-track delta-time reconstruction (unchanged). Chords are keyed by original
  absolute tick across tracks; a coupled member on another track targets the anchor's
  post-clamp landing, clamped to its own track's window. For a single-track file the
  merged order equals the original order, so output is **byte-identical** (verified
  against the pre-refactor engine across phi/push/intensity/timing-only/velocity-only
  with the real rock.json). Multi-track same-tick hits now land ≤2 ticks apart at
  phi=0.5 (was up to ~73 ms flams; regression test
  `TestKitWideClock::test_cross_track_same_tick_hits_land_together`). Note: for
  multi-track files the RNG sample order changes from track-major to time-major, so
  multi-track output differs from the old engine at ANY phi, including 0 — inherent
  to stepping one clock in performance order.

## Implementation notes — module 13: velocity rebuild (schema v2)

Spec: `wobblemidi_rebuild_spec.md` (v2) + `wobblemidi_rebuild_spec_addendum.md`. Shipped
2026-07 with `wobblemidi/profiles/rock.json` rebuilt from the FULL GMD (341 rock takes,
311 buckets). Gated on held-out data by `scripts/validate.py` (see below), then ear-tested.
Old-schema profiles still load and run (backward compatible; their vel_delta bias is
applied statically, never amplified).

**A2 — `vel_delta` redefinition (the core fix):** residual vs the shrunk
`(take, grid_position, instrument[, tier])` mean. Accent structure lives in the contour
the USER programs; the v1 bucket-median delta sampled it as noise (hats jumped soft→127).
Residual σ roughly halved (snare 20.7 → 9.9 with tier conditioning; kick 14.1, hats 20.8).
Two REQUIRED guardrails, both asserted at build time:
- **Per-emitted-bucket de-bias to mean ~0** (`_write` asserts it): residualising at cell
  level does NOT zero tier-bucket means — real tier buckets carried biases up to ±28
  pre-debias. Removed bias stored in `_meta.bucket_vel_delta_means` (diagnostics).
- **Sparse-cell shrinkage** (`SHRINKAGE_K = 5` pseudo-counts): an n=1 cell would produce
  a fake zero-noise residual (lone crashes/fills). Cells shrink toward the broader
  `(take, instrument)` mean; n ≫ 5 cells are essentially untouched.

**Snare tier-residualisation (addendum Fix 1):** snare alternates ghost/backbeat at ONE
grid position across bars, so the tier-agnostic cell mean lands between the roles and the
residual re-absorbs accent structure — snare was the one instrument that regressed at
checkpoint 3. Tier is a per-take relative ROLE label via `_file_tier_thresholds` (the SAME
convention runtime B4 and the harness use; GMD-absolute thresholds are the per-take
evidence fallback). Shrinkage chain is TIER-PRESERVING: `(take,pos,inst,tier)` →
`(take,inst,tier)` → `(take,inst)` — never the blended positional mean (that blend was the
bug). **Snare only** (`TIER_RESIDUAL_GROUPS`): per-take two-cluster rate is 10% for snare
vs 1–3% for kick/hats/ride. Do NOT extend by the naive variance-reduction measure — it is
confounded (slicing pure noise by velocity-derived roles gives ~82% reduction; the
mechanical null).

**B2 — kick-only velocity drift:** a second `GrooveDrift` instance (`PHI_VEL = 0.37`,
calibrated: train-split kick velocity lag-1 r=+0.317 / (1−RESIDUAL_SHARE) —
`scripts/calibrate_phi.py` now has a velocity section and `--train-only`). Kick only:
snare/hats/ride/crash measured ~white (r 0.07–0.16); no cross-instrument sharing — a
same-tick crash/snare keeps its own draw. Runs on a THIRD dedicated RNG stream
(`vel_seed = seed + 1_779_033_703`), which preserves both existing contracts: samples
identical across phi, and timing identical across timing_only/default. Centred on the
bucket's own vel_delta mean (legacy-profile bias applied statically — same guard as the
timing clock). `timing_only` does not step the vel clock. The module-12 "phi=0 exact
bypass" contract is now TIMING-only: velocities take the new path at all phi.

**B4 — relative velocity tiering (fixes tier-collapse):** per-file tier thresholds from
the user's own per-instrument velocities (`_file_tier_thresholds`), passed into `_lookup`
as a thresholds override — the tier selects the TIMING bucket too. Rules: relative only
with evidence (n ≥ 8 hits AND p90−p10 ≥ 12); exactly two distinct values → soft/hard at
their midpoint regardless of balance (an imbalanced 14-ghost/2-accent part must not fall
through to tertiles that collapse onto the dominant value); two-cluster parts → soft/hard
at the dominant gap's midpoint, only when the gap ≥ 1.5× either side's spread AND both
sides hold ≥ 15% of hits (a 3-level evenly-spaced part must get tertiles, not soft/hard);
otherwise relative tertiles with a tie-aware re-anchor (thresholds that collapse onto a
dominant duplicated value move to midpoints BETWEEN distinct values); absolute fallback
otherwise; ties always share a tier.

**A4 — storage contract unchanged:** values stay `[[offset_ms, vel_delta], ...]`; only
`_meta` grew: `schema_version: 2`, `vel_delta_definition`, `bucket_vel_delta_means`,
`vel_sigma_within`, `tier_residual_groups`. `test_shipped_profile_is_schema_v2` pins the
shipped artifact (never ship an old-schema rebuild again). Build CLI gained
`--split all|train` and `--output` (gate/candidate builds never touch the bundled file).

**O5 — test locks retired:** the `(1-RESIDUAL_SHARE)*phi` autocorr test and the
AR-recursion-constant test are gone, replaced by invariants: autocorr monotone in phi
(unit + through-engine), phi=0 → no timing memory, total timing-variance budget across
phi, seed determinism, per-bucket velocity de-bias ≈ 0. Do not reintroduce
exact-constant locks — they fight every recalibration.

**Validation harness (`scripts/validate.py`, spec Part C):** gates profile rebuilds on
held-out GMD (take-level split: gate profile from train, eval on test). Programmed inputs
= 16th-grid quantise + 4-level per-instrument velocity palette per take (2/8/flat as
sensitivity). Metrics per instrument with multi-seed CIs: offset Wasserstein/KS/σ/mean,
velocity within-position σ, adjacent-jump distribution, contour preservation (per-position
MAE + hit-matched Spearman), lag-1 autocorr (timing pooled de-lean per calibrate_phi;
velocity de-meaned per (take, instrument, position)), same-slot cross-instrument gap σ,
and the anti-robotic pair from the addendum: zero micro-jump mass (|Δv| ≤ 1 fraction) and
within-(position, role) σ with FIXED human-derived role labels — both scored TWO-SIDED
against human, never "beat the input" (the input is a coarsened photocopy of the answer
and wins any naive distance). The harness pins `intensity=1.0` — it measures the engine's
full-scale reproduction of human distributions, so recorded baselines stay comparable
regardless of the product default.

**Known open items (measured, deliberately not tuned):** within-role velocity over-noise
at full intensity (hats ~20.8 / ride ~17.5 / kick ~14.1 vs human ~6.4–8.5 within a role;
snare fixed to gate level) — masked at the 0.35 default; snare zero-jump mass marginally
under the gate (continuous KDE residuals rarely produce the exact velocity repeats real
snare ghosting has).

**Ear-test kit:** `scripts/make_eartest.py` — `rock_ghosts` (snare ghosts + busy 16th
hats) and `four_floor` patterns; old/new renders, velocity-only / timing-only diagnostic
legs (the isolated RNG streams make the decomposition exact), and a timing-only intensity
sweep. Historical A/Bs: extract an old profile via
`git show <commit>:wobblemidi/profiles/rock.json` and pass `--old`.

## Implementation notes — default intensity 0.35 (was 1.0)

Default/label change ONLY — no engine, profile, or timing-model change (commit d7cf0b6).

**Ear-tested rationale:** the timing-only intensity sweep (0.3/0.5/0.7, same seed = same
draws scaled linearly) isolated the "jagged" complaint about the rebuilt output: 0.3
sounded good, 1.0 sloppy. 1.0 faithfully reproduces GMD's raw within-take spread (~27 ms
timing σ) — real drummers genuinely play that loose (roadmap measured fact: "sloppy at
1.0 is taste, not a variance bug"). 0.35 ≈ σ 10 ms on a 95 BPM pattern; for reference,
0.5 ≈ the tighter half of GMD takes (σ ~17 ms at P10), 0.3 is tighter than nearly any
human take in the corpus.

Key decisions:
- Range stays `FloatRange(0.0, 1.0)`, NOT hard-capped; CLI help + README steer:
  "0.2–0.5 is the useful range; higher values reproduce raw drummer spread and will
  sound loose."
- `scripts/validate.py` pins `intensity=1.0` explicitly — gating is a full-scale
  engine-vs-human comparison and must not silently move with the product default.
- Ten mechanics tests pin `intensity=1.0` explicitly (they test engine behaviour, not
  the default); `test_defaults` locks 0.35.

## Implementation notes — lean + per-group intensity (GUI round 3, engine params)

Two `humanise()` params added for the GUI (2026-07-04). Both are per-hit arithmetic
applied AFTER `_sample_bucket()` — RNG streams and every module-12/13 contract hold by
construction; byte-equality at the defaults is test-locked (`TestLeanAndPerGroupIntensity`).

**`push_amount: float | None = None` (lean).** Generalises the `push` bool:
`offset - (0.0 if push else mu)` became `offset - (1.0 - lean) * mu` at both de-bias
sites (phi==0 bypass + solo path), `lean = push_amount if not None else (1.0 if push
else 0.0)`. Mutually exclusive with `push=True` (raises); range [-1, 1] validated.
HONEST SCOPE: -1 MIRRORS the stored per-bucket means — it inverts the source drummers'
tendencies, it is NOT a synthetic "laid-back drag" (the shipped profile is not uniformly
early; ride leans late). UI labels the negative side accordingly. Known interaction: the
de-bias precedes the intensity multiply, so injected lean scales with the group's eff —
"tight but pushing" is not expressible. Legacy (no-means) profiles: any lean is a no-op;
an explicit `push_amount != 1.0` prints a one-line stderr note (test-locked no-op).

**`intensity_by_group: dict[str, float] | None = None`.** ABSOLUTE per-lane
humanisation amount (output gain on that lane's deviation):
`eff(group) = dict.get(group, intensity)`. Keys validated against `TD11_TO_GROUP`
values (unknown → ValueError, not silent); values must be >= 0. HONEST SCOPE — this is
NOT independent per-lane timing feel:
- The kit still shares ONE drift clock: `groove.step` consumes the UNSCALED centred
  sample, so a 0.0 lane still drives the shared drift (test-locked: kick output is
  identical whether hats are at 0.9/0.1/unset on a collision-free pattern).
- Same-tick chords are governed by the TIGHTEST limb: the anchor's candidate is scaled
  by the per-tick MIN eff across all shiftable hits at that original tick
  (`tick_min_eff`, precomputed in pass 1, order-independent — locked in both stream
  orders); coupled members scale their ±1 ms residual by their OWN eff, so a 0.0 member
  sits exactly on the anchor's landing.
- `_new_velocity` uses `eff(group)`; eff=0 → velocities untouched.
GUI exposes this as lane-select (INTENSITY knob scoped to a clicked lane); TIGHTNESS and
LEAN stay kit-wide in the UI to match the engine truth. A per-lane fader bank (808
style) is the noted future UI if per-drum use expands. CLI flags for both params:
deferred.

## Implementation notes — time-windowed chord coupling (COUPLE_WINDOW_MS)

Codex-design-reviewed; shipped 2026-07-05. Generalises same-tick coupling outward by a
small REAL-TIME window so close-spaced ornaments (snare flams, grace notes) move as one
rigid unit instead of scattering. Safety, not drama — subtle or nothing.

- **Membership:** consecutive shiftable hits whose elapsed time from the cluster's FIRST
  hit is `<= COUPLE_WINDOW_MS` (12.0, module constant, ear-tunable, NOT a CLI knob),
  measured via `ticks_to_ms_with_map` (PPQ alone is not time). Inclusive boundary,
  test-locked in ms at 1 tick = 1 ms tempo.
- **Only gap>0 clusters take the new path.** Same-tick (zero-gap) clusters keep the
  existing anchor+residual behaviour BYTE-FOR-BYTE, singletons stay solo, phi==0 still
  disables all coupling — locked by `test_pre_round3_fixture_regression` against real
  pre-round-3 engine output (fixtures from commit 63862a1^, recipe in the test docstring).
- **One rigid shared tick delta** per cluster: sourced from the LOUDEST member's sample
  (main stroke leads, grace follows; velocity ties → earliest in merged order), scaled by
  `cluster_min_eff` (min `_eff` across ALL members, computed before emission — deliberately
  NOT the exact-tick `tick_min_eff`), clock stepped ONCE on the loudest member's centred
  sample. No per-member residuals.
- **Cluster-scope clamp (the anti-wonk guarantee):** each member's legal delta interval
  vs its own fixed context (own note_off, next fixed event, prior emitted state, with
  intervening fixed events replayed via a small simulation) is INTERSECTED and the one
  shared delta clamped once. Members are never clamped independently — that is what would
  collapse a flam. Member-vs-member spacing needs no constraint: the shared delta
  preserves original gaps (>= EPSILON_TICKS). Empty intersection → delta 0 (hold as
  written).
- **RNG contract preserved:** members' samples are drawn eagerly IN MERGED ORDER at the
  first member's turn (members are consecutive shiftable hits, so the global stream is
  unchanged); velocities still come from each member's own sample at its own loop turn
  (vel clock order untouched). Files without gap>0 clusters are byte-identical to before.
- Ride-alongs in the same commit: `intensity_by_group` finite-value guard (nan passed a
  bare `v < 0`), 3-lane MIN-eff lock, pre-round-3 fixture regression replacing the
  self-comparing push-endpoint test, −1 MIRROR locked on the phi==0 bypass too.

**P1 follow-up (elastic member offs):** a member's OWN note length must never bound the
shared delta — a 1-tick hat is a normal drum note, not a timing wall (Codex repro: h_hi=1
from the hat's length vs h_lo=4 from a late prior kick → per-member fallback → 10→6
smear). Rigid-cluster member note_offs are now ELASTIC: emitted at
`max(written, prev_emitted)`, i.e. they ride behind their shifted on (sliding forward
only when the on would otherwise pass them — may truncate a 1-tick note to 0 length in
the squeezed hold case; inaudible at any real PPQ). Both planner intervals use real
WALLS only (`_next_wall`: first event that is neither shiftable nor a member off);
off-side slide is bounded via its driver on (`wall(off) − last member-on before it`),
never via the note's duration. The per-member fallback now needs a late prior hit AND a
foreign fixed wall inside the window simultaneously (no shared delta is encodable at
all in that corner — rigid or not — so it cannot be removed without trading smear for a
crash); it is unreachable from note durations alone. Solo hits keep their own-off
ceiling (unchanged semantics outside clusters).
