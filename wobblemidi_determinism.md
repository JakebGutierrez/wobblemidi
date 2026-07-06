# wobblemidi — determinism contract

_Frozen 2026-07-06 (Fable window, Session 3). This document is one of the three pillars of
the porting contract (with `wobblemidi_porting_contract.md` and the golden vectors under
`tests/golden/`). It states exactly what reproducibility the engine guarantees, where every
random draw comes from, and every float→integer boundary in the output path. All line
references are to the engine at the commit this file was introduced; `tests/golden/`
byte-locks the behaviour itself, so the vectors — not this prose — are the arbiter._

## The guarantee

Given identical **input MIDI bytes**, identical **profile JSON bytes**, identical
**parameters**, and an **explicit integer `seed`**, `humanise()` produces **byte-identical
output**, at the following confidence levels:

| Scope | Guarantee |
|---|---|
| Same process, repeated calls | Byte-identical. Each call reseeds all three streams from `seed`; no state leaks between calls (the polar-method spare-gaussian cache is reset by reseeding). |
| Same machine, separate processes | Byte-identical, regardless of `PYTHONHASHSEED` — no set/dict iteration order reaches the output path (see "Ordering"). |
| Different OS/arch, same numpy + scipy versions | Byte-identical expected. numpy's legacy `RandomState` stream is version- and platform-frozen (NEP 19); all tick arithmetic is integer or IEEE-754 basic ops. The one caveat: `RandomState.multivariate_normal` (inside KDE resampling) factorises the bucket covariance via LAPACK SVD, so a different BLAS can differ in the last ULP — which only matters if a value lands exactly on a `round()` half-integer boundary. Checked empirically, not proven: golden vectors are generated on macOS arm64 / CPython 3.14.4 / numpy 2.5.1 / scipy 1.18.0 and re-verified by CI on ubuntu x86-64 / CPython 3.11 on every push. |
| Across numpy/scipy versions | `RandomState` variate streams are frozen by NEP 19. scipy's `gaussian_kde.resample` algorithm is stable in practice but not contractually frozen by scipy. A golden-vector CI failure after a dependency bump is **drift detection working** — treat it as signal, diagnose, and only then regenerate vectors deliberately. |

**Explicitly not guaranteed:**

- `seed=None` is **non-deterministic by design** — it reseeds all three streams from OS
  entropy (`humanise.py:450,453,459`). The contract requires an explicit seed. The CLI
  exposes `--seed`; the GUI adapter always passes one.
- Thread safety. Stream 1 is the **process-global** `np.random` singleton: `humanise()`
  owns it for the duration of a call. Two interleaved `humanise()` calls, or any other
  library consuming `np.random` concurrently, break reproducibility. Single-threaded
  contract. (Parked hardening, **not** done in this window and gated on engine review:
  pass an owned `RandomState` into `kde.resample(seed=...)` and the `randint` fallback —
  in principle byte-preserving since it is the same MT19937 sequence.)
- Bit-exact KDE fits across BLAS builds (see table caveat above).

## The three RNG streams

All three are derived from the single user-facing `seed` at the top of `humanise()`
(`humanise.py:450-460`):

| Stream | Construction | Consumed by | Purpose |
|---|---|---|---|
| 1 — sample stream | `np.random.seed(seed)` (global singleton) | `_sample_bucket()` only (`humanise.py:303-321`) | The (offset_ms, vel_delta) sample per hit: `bucket.kde.resample(1)` (no `seed=` argument → global singleton), or `np.random.randint(len(bucket.offsets))` uniform-index fallback for degenerate (`kde=None`) buckets. KDE draws are clamped to the bucket's own min/max (`:316-317`). |
| 2 — timing-clock residuals | `RandomState((seed + 2_246_822_519) % 2**32)` | The kit-wide timing `GrooveDrift` (`:623`), one `normal(0, sigma)` per `step()` (`:383`) | AR(1) drift residuals. Constructed only when `phi != 0.0`. |
| 3 — velocity-clock residuals | `RandomState((seed + 1_779_033_703) % 2**32)` | The kick velocity `GrooveDrift` (`PHI_VEL=0.37`, `:460`), via `_new_velocity` (`:518-521`) | Kick-only velocity drift residuals. |

The derivation constants are arbitrary fixed 32-bit offsets; the `% 2**32` keeps the
derived seeds in `RandomState`'s accepted range. They are part of the contract — a port
targeting Tier 1 must reproduce them exactly.

**Why three streams — the separation contracts.** These are load-bearing, test-locked
invariants, not implementation accidents:

- Stream 1 is consumed identically at every `phi` and in every mode → a phi A/B at one
  seed is a clean timing-only comparison; `phi=0` and `phi=0.9` draw the *same samples*.
- Streams 2 and 3 are separate from each other → timing is identical across
  `timing_only`/default (the velocity clock never perturbs timing draws), and velocities
  are identical across `velocity_only`/default at the same seed. This is what makes
  `scripts/make_eartest.py`'s diagnostic decomposition exact.

## Draw inventory (execution order)

Processing order is the **merged time-major stream**: all messages across all tracks,
stable-sorted by absolute tick (see "Ordering"). Per event:

| Event | Stream 1 | Stream 2 | Stream 3 |
|---|---|---|---|
| Shiftable hit (every mode, every phi) | **Exactly 1 `_sample_bucket` call**, in merged order | — | — |
| Solo hit / chord anchor (`phi>0`, not `velocity_only`) | (its 1 sample) | 1 `normal` (`:919`) | — |
| Rigid windowed cluster, at its first member's turn (`phi>0`) | Remaining members' samples drawn **eagerly in merged order** (`_plan_cluster`, `:695`) — still exactly 1 each, stashed and popped at each member's own turn (`:820-825`) | 1 `normal` for the whole cluster, on the loudest member's sample (`:710`) | — |
| Coupled same-tick member (`phi>0`) | (its 1 sample) | **0** — the ±1 ms residual is deterministic from its own sample (`:901-903`); does not step the clock | — |
| Kick hit whose velocity is humanised (default or `velocity_only`) | — | — | 1 `normal` (`:520`) |
| Non-kick velocity, `timing_only` velocities, non-shiftable events | — | — | — |
| `phi == 0.0` | 1 sample per shiftable hit, unchanged | 0 (clock never constructed, `:623`; window scan skipped, `:636`) | kick normals unchanged |
| `velocity_only` | 1 sample per shiftable hit, unchanged | 0 (`continue` at `:835` precedes all timing) | kick normals unchanged |
| `timing_only` | 1 sample per shiftable hit, unchanged | per-phi as above | 0 (`:845-846`, `:928-929`) |

**Variate-count nuance.** "One `_sample_bucket` call" is the API-level unit. The number of
underlying MT19937 variates differs by path — a KDE resample consumes several (see the
porting inventory), the `randint` fallback consumes a bounded-integer draw — but the path
taken depends only on the bucket (profile data), which is a fixed input. Mode and phi never
change which path a given hit takes, so the stream stays aligned across all A/B contrasts
above.

Everything else in the engine is RNG-free: `_lookup`, `_file_tier_thresholds`
(percentiles), `_ms_offset_to_ticks`, all windowing/clamping, and `load_profile` (KDE
*fitting* is deterministic given the bucket arrays; only *resampling* draws). The profile
build (`scripts/build_profiles.py`) contains no RNG at all — `rock.json` is a deterministic
function of the GMD dataset.

## Ordering guarantees

- The master ordering is `merged.sort(key=lambda e: e[0])` (`humanise.py:612`) — Python's
  stable Timsort on absolute tick only, so ties keep (track index, message index) order
  from construction (`:607-611`). For a single-track file the merged order equals the file
  order (byte-identity with the pre-kit-wide engine is test-locked).
- `build_tempo_map` (`midi_utils.py:58`) and `detect_meter` (`:201`) sort events by
  absolute tick, stable; a missing tempo at tick 0 inserts the 500 000 µs default
  (`:59-61`).
- **No set/dict iteration order reaches the output.** The one set iterated in an output
  path is `member_offs` (`humanise.py:772-778`), and its loop body only performs `min()`
  reductions into `d_hi`/`h_hi` — commutative and order-independent (and int-tuple hashing
  is not randomized by `PYTHONHASHSEED` anyway). All other sets are membership tests or
  error-message formatting; all dicts are keyed lookups. Verified empirically: the golden
  suite passes under different `PYTHONHASHSEED` values.
- Note-on/off pairing is FIFO per `(note, channel)` via deques (`humanise.py:580-592`) —
  deterministic.

## Float → integer boundaries (complete inventory)

Every place a float becomes an output tick or velocity. A Tier 1 port must match these
exactly; Python `round()` is **round-half-to-even** (banker's rounding), which differs
from C's `round()` (half-away-from-zero).

| Site | Rule |
|---|---|
| `quantise_to_grid` (`midi_utils.py:139-144`) | Integer-only: `subdivision = ppq // 4` (16th grid) or `ppq // 2` (8th, 6/8 files); snap down iff `remainder < subdivision // 2` — an exact midpoint snaps **up**. |
| `grid_position_in_bar` (`midi_utils.py:153-155`) | Integer-only: `(grid_tick % (16 * (ppq // 4))) // (ppq // 4)`. |
| `_ms_offset_to_ticks` (`humanise.py:249-300`) | Piecewise walk over the tempo map; within a segment, `round(remaining / ms_per_tick)` (**half-to-even**) at `:273,:279,:292`; `ms_per_tick = tempo_us / ppq / 1000.0`; loop runs while `remaining > 1e-9`; backward walk clamps at tick 0 (`:297-298`) and steps off exact boundaries to avoid a zero-width segment (`:284-285`). |
| Solo/coupled placement (`humanise.py:950`, escape hatch `:873`) | `int(max(lower, min(candidate, ceiling)))` — `candidate` is already integer (grid tick + integer delta); the clamp bounds are integers or `math.inf`. |
| Cluster shared delta (`humanise.py:782`) | `int(max(d_lo, min(float(desired), d_hi)))` — `int()` **truncates toward zero** when `desired` is fractional and unclamped. Hold fallback: `int(max(0.0, h_lo))` (`:794`). |
| Velocity (`humanise.py:524`) | `round(velocity + vel_delta * eff)` (**half-to-even**), then clamp to `[1, 127]`. |
| Coupling-window membership (`humanise.py:660-661`) | `ticks_to_ms_with_map(first, e) <= COUPLE_WINDOW_MS` (12.0), **inclusive**; ms accumulate as floats segment-by-segment in tempo-map order (`midi_utils.py:107-118`). |
| Minimum separation | `EPSILON_TICKS = 1` (`humanise.py:28`), integer arithmetic throughout. |

`mido.save()` embeds nothing environment-dependent: identical message lists produce
identical bytes.

## Porting inventory (Tier 1 only)

Byte-matching the golden vectors (**Tier 1** of the port correctness definition in
`wobblemidi_porting_contract.md` — optional; Tier 2 is the sufficient gate) requires
reimplementing, exactly:

1. **MT19937** and numpy's *legacy* `RandomState` variate algorithms at numpy 2.5.1
   (`numpy/random/mtrand.pyx` — frozen by NEP 19): `standard_normal` (Marsaglia polar
   method, draws uniform pairs with rejection and **caches the spare gaussian across calls
   on the same state**), `randint` (masked rejection), `choice(n, size, p=...)`
   (cumsum-of-p + searchsorted inversion of uniforms), `multivariate_normal`
   (standard normals transformed through an **SVD** factorisation of the covariance).
2. **`scipy.stats.gaussian_kde`** at scipy 1.18.0: fit (per bucket, at profile load) sets
   `self.covariance = data_covariance × factor²` with Scott's factor `n**(-1/(d+4))`,
   d=2; resample draw order verified from source —

   ```python
   norm    = transpose(random_state.multivariate_normal(zeros(d), self.covariance, size=size))
   indices = random_state.choice(self.n, size=size, p=self.weights)   # weights uniform here
   return  self.dataset[:, indices] + norm
   ```

   i.e. per sample: the 2-D gaussian noise draw **first**, then the weighted component
   pick. Both on stream 1.
3. The seed derivations (`+ 2_246_822_519` / `+ 1_779_033_703`, mod 2³²), the draw
   inventory above, the merged stable sort, and the float→int rules above (half-to-even
   `round`, truncating `int()` at the cluster sites, floor-division grid math).

Consult the pinned library sources rather than this summary when in doubt — and let the
vectors arbitrate.

## Verification artifacts

- `tests/golden/` + `scripts/verify_golden.py` — the byte-locks themselves (~26 vectors
  over 10 fixtures sweeping the parameter surface); run in CI via
  `tests/test_golden_vectors.py`.
- Existing locks: `test_same_seed_same_output` (`tests/test_humanise.py`), CLI
  `test_seed_determinism` (`tests/test_cli.py`), and the historical byte-fixtures
  `tests/fixtures/pre_r3_*.mid`.
- CI (ubuntu x86-64, CPython 3.11, unpinned pip resolve) re-verifies the vectors generated
  on macOS arm64 / CPython 3.14 — the standing cross-platform, cross-version check.
