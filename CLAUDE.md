# pocketmidi — Claude Code context

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
| 2 | `pocketmidi/midi_utils.py` | done |
| 3 | `scripts/build_profiles.py` | done |
| 4 | `pocketmidi/humanise.py` | done |
| 5 | `pocketmidi/cli.py` | done |
| 6 | `tests/test_humanise.py` | done |
| 7 | `--timing-only` / `--velocity-only` flags | done |
| 8 | Velocity-stratified buckets + KDE sampling | done |
| 9 | Grid position awareness | done |
| 10 | Outlier clipping | done |
| 11 | 6/8 support | done |

Build one module at a time. Use plan mode for each new module.

## Workflow
- Plan mode before any multi-file or new-module work
- Read existing code before editing
- Commit after each working module
- Never add Co-Authored-By lines to commit messages

## Design decisions — do not change without discussion

**Instrument mapping:** Roland TD-11 only. Notes 22 and 26 are hi-hat edge
variants not in the GM spec — they must stay in `TD11_TO_GROUP`. See
`pocketmidi/midi_utils.py`.

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
`vel_delta`: delta from median velocity for that bucket, not raw velocity.

**Sparse fallback order:**
1. `(genre, beat_type, instrument)` — exact
2. `(genre, "beat", instrument)` — drop fill context
3. `("global", instrument)` — pooled
4. no change applied

**Ghost note filter:** `VELOCITY_FLOOR = 20` for kick and snare only during
profile build. Hi-hats and cymbals are exempt.

**Intensity:** Scales sampled deltas linearly toward zero —
`applied = sampled * intensity`. Do not clamp before scaling.

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
Output: `pocketmidi/profiles/rock.json`

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

Requires rebuilding `pocketmidi/profiles/rock.json`. Batch items 1 + 3 together —
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
Entry point: `pocketmidi <input.mid> <output.mid>`

Non-obvious implementation decisions:
- **Profile resolution:** Genre maps to `pocketmidi/profiles/{genre}.json` via
  `importlib.resources.files("pocketmidi.profiles").joinpath(...)` + `as_file()`.
  `as_file()` is required (not `str()`) to guarantee a real filesystem path in all
  install layouts (editable, wheel, zip-imported).
- **`--section` flag:** User-facing name for `beat_type` — maps directly to the
  `beat_type` parameter of `humanise()`.
- **`--intensity` validation:** Uses `click.FloatRange(0.0, 1.0)` — Click rejects
  out-of-range values before `humanise()` is called.
- **Packaging:** `[tool.hatch.build] include` covers both wheel and sdist so
  `pocketmidi/profiles/*.json` ships in all distribution formats.

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

4/4 gate in `humanise()` is conditional: only fires when the loaded profile
contains grid-pos buckets (detected by `key.split("|")[-1].isdigit()`). Legacy
profiles without grid-pos keys skip the check and work on any time signature.

rock.json rebuilt from GMD: 315 buckets (277 grid-position, 38 legacy fallback),
7 non-4/4 files skipped.

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
- The 4/4 gate now uses `meter != "6/8"` to bypass for 6/8 files — they are accepted
  even with grid-pos profiles (though grid_pos=None means no positional lookup).
- Both note-processing loops pass `grid` to `quantise_to_grid` and compute
  `gp = None if meter == "6/8" else grid_position_in_bar(grid_tick, ppq)`.
  Passing `grid_pos=None` skips the positional bucket chain entirely and falls back
  to the per-instrument ms deviation buckets, which transfer well across meters.

**No profile rebuild needed.** The existing rock.json is reused unchanged.
