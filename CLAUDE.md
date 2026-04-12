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

**Grid:** Straight 16th-note grid only. No swing/triplet in v1.

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

2. **`--timing-only` / `--velocity-only` flags** — apply timing humanisation without
   touching velocity, or vice versa. Useful when input already has good velocity
   variation (e.g. finger-drummed then quantised).

3. **KDE sampling** — replace flat independent sampling with
   `scipy.stats.gaussian_kde`. Storage format already supports this as a drop-in.
   Do this alongside velocity stratification (both require rebuilding profiles).

4. **Grid position awareness** — bucket key becomes
   `(genre, beat_type, instrument, grid_position)`. Beat 1 kick vs off-beat kick
   have different timing tendencies in real drumming.

5. **Custom profile source** — `--profile path/to/custom.json` flag so users can
   build profiles from their own MIDI packs (e.g. professional drummer sample packs)
   and humanise to sound like a specific player.

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
