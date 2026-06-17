# pocketmidi — Claude Code context

## Review style
When asked to review code or a plan, output only blocking issues — incorrect 
logic, edge cases that would corrupt output, or broken contracts between 
modules. Ignore style, formatting, and minor suggestions unless explicitly asked.

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
| 10 | Outlier clipping (KDE tail fix) | done |
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

**Time signature:** Auto-detected via `detect_meter()` in `midi_utils.py`. 6/8 uses
an eighth-note grid and skips grid-position lookups (`grid_pos=None`). Files mixing
6/8 with other signatures are rejected. 3/4 and other uniform quarter-note meters use
the 16th grid unchanged.

**MIDI file type:** Type 0 and type 1 only. `humanise.py` builds a single song-level
tempo map and applies it across all tracks. Type 2 (independent per-track timing) is
not supported — GMD files are type 0/1.

**Genre filter:** Rock only in v1. Filter: `df[df["style"].str.startswith("rock")]`.
Jazz/funk profiles are explicitly out of scope (swing feel is misrepresented under
a straight 16th grid).

**Bucket key:** `(genre, beat_type, instrument_group, grid_position)` — grid_position is 0–15 (16th-note index in a 4/4 bar). See module 9 notes in CLAUDE.md.

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

1. **Velocity-stratified buckets** — done. kick tertiles 57/80, snare 53/118.

2. **`--timing-only` / `--velocity-only` flags** — done.

3. **KDE sampling** — done. `scipy.stats.gaussian_kde` fitted at load time per bucket.

4. **Grid position awareness** — done. See module 9 notes below.

5. **Custom profile source** *(deferred — out of scope for now)* — `--profile
   path/to/custom.json` flag so users can build profiles from their own MIDI packs
   (e.g. professional drummer sample packs) and humanise to sound like a specific
   player. Revisit if there is a concrete use case requiring a non-GMD source.

6. **6/8 support** — done. See module 11 notes in CLAUDE.md.

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

## Implementation notes — completed modules (module 8)

### Module 8: velocity-stratified buckets + KDE sampling
Done. rock.json rebuilt: kick tertiles 56/79, snare tertiles 52/117. 38 buckets.

- `build_profiles.py`: stratified tier buckets (`rock|{beat_type}|{instrument}|soft/medium/hard`)
  for kick and snare only; thresholds from post-filter tertiles; `_meta` key carries
  `velocity_thresholds` and `kde_bw_method` (default `"scott"`)
- `humanise.py`: `BucketProfile` / `LoadedProfile` dataclasses; `load_profile` validates
  `kde_bw_method` and fits 2D KDE per bucket at load time; `_lookup` uses 4-level fallback
  for kick/snare (exact tier → drop tier → drop fill → global), 3-level for all others;
  `_sample_bucket` replaces dual index draws with `kde.resample(1)`; degenerate buckets
  keep `kde=None` and fall back to uniform pair sampling

**Bandwidth tuning:** change `KDE_BW_METHOD` in `build_profiles.py`, rebuild profile.
`load_profile` reads it from `_meta` — no code changes to `humanise.py` needed.

**KDE bandwidth — check by ear after first rebuild.** Scott's rule is the default.
Hi-hat timing can be bimodal (on-grid and behind-the-beat clusters) — Scott's rule
may over-smooth this into one blob, making the hi-hat feel smeared rather than
pocketed. Listen to a hi-hat pattern after rebuild and adjust bandwidth in
`build_profiles.py` if needed. Not a CLI flag.

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
- Returns `"6/8"` only if every `time_signature` event is 6/8 AND the first is at
  tick 0 (no implicit 4/4 prefix).
- Returns `"non-6/8"` for no events (MIDI default 4/4), uniform 4/4, uniform 3/4,
  and non-6/8 mixed-meter files (16th grid is valid for all quarter-note-based meters).
- Raises `ValueError` if 6/8 is mixed with any other signature, or if the first 6/8
  event is not at tick 0.

**`quantise_to_grid(time_ticks, ppq, grid="16")`** — `"8"` uses `ppq // 2`.
Default `"16"` unchanged; all existing callers unaffected.

**`humanise()` changes:**
- Calls `detect_meter(mid)` after the type-2 check; sets `grid = "8"` for 6/8.
- 4/4 gate bypassed for 6/8 files (`meter != "6/8"`).
- Both note-processing loops pass `grid` to `quantise_to_grid` and set
  `gp = None if meter == "6/8"` — skips positional lookup, uses per-instrument
  ms deviation buckets, which transfer well across meters.

No profile rebuild needed.

## Implementation notes — --push flag / offset de-bias

**Problem:** GMD rock drummers push kick ahead of the beat at certain grid positions.
At low intensity this creates a systematic early lean that sounds like rushing.

**Design:** Per-bucket mean offsets stored in `_meta.bucket_offset_means` at build time.
De-bias applied by default at sample time: bucket mean subtracted from `offset_ms_raw`
before intensity scaling. `--push` restores original GMD behaviour.

Key decisions:
- `_clip_hits()` lets `_build_profiles()` access the retained set directly to compute
  the mean, without changing `_build_pairs_with_clip()`'s return contract.
- `_build_profiles()` returns `(profiles, bucket_offset_means, written, skipped)`.
- `LoadedProfile.bucket_offset_means` has `default_factory=dict` — existing call
  sites unchanged; old profiles without the key get 0.0 correction (backward compat).
- `_lookup()` return extended to `(BucketProfile | None, int | None, str | None)`;
  matched key used for mean lookup; level unchanged.
- `_build_pairs` and `_sample_bucket` unchanged.
- Requires profile rebuild to populate `bucket_offset_means` in `_meta`.
