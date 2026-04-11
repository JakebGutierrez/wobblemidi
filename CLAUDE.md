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
| 3 | `scripts/build_profiles.py` | next |
| 4 | `pocketmidi/humanise.py` | pending |
| 5 | `pocketmidi/cli.py` | stub only |
| 6 | `tests/test_humanise.py` | pending |

Build one module at a time. Use plan mode for each new module.

## Workflow
- Plan mode before any multi-file or new-module work
- Read existing code before editing
- Commit after each working module

## Design decisions — do not change without discussion

**Instrument mapping:** Roland TD-11 only. Notes 22 and 26 are hi-hat edge
variants not in the GM spec — they must stay in `TD11_TO_GROUP`. See
`pocketmidi/midi_utils.py`.

**Grid:** Straight 16th-note grid only. No swing/triplet in v1.

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

## Implementation details — pending modules

### build_profiles.py
- Reads `info.csv` from the unzipped GMD directory; key columns: `midi_filename`, `style`, `beat_type`
- Filter rock: `df[df["style"].str.startswith("rock")]`
- `MIN_SAMPLES = 30` — buckets below this threshold are not written to the profile; the runtime fallback handles them
- `VELOCITY_FLOOR = 20` — drop hits below this for kick and snare only during ingest (ghost note / accidental trigger suppression); hi-hats and cymbals are exempt
- Takes path to unzipped `groove-v1.0.0/` as CLI argument

### humanise.py
- Pre-compute marginal arrays **at profile load time**, not per-hit:
  ```python
  offsets = np.array([p[0] for p in pairs])
  vel_deltas = np.array([p[1] for p in pairs])
  ```
- Filter `note_on` with `velocity=0` — these are note-offs in disguise, not real hits
- Dense hit safety: `new_time = max(prev_event_time + epsilon, new_time)`
  — `prev_event_time` is the timestamp of the last emitted **note-on**, not note-end (drum note lengths are artificial)
- Clamp output velocity to 1–127; never produce negative delta times; never reorder notes
- `np.random.seed(seed)` — seed applies to numpy random state and affects both timing and velocity sampling

### cli.py
- Entry point: `pocketmidi <input.mid> <output.mid>`
- Flags: `--genre rock`, `--intensity 0.7`, `--section beat|fill` (default: `beat`), `--seed 42`, `--verbose`
- `--verbose`: log which fallback level was used per hit
