# wobblemidi

**Humanise programmed drum MIDI with the timing and velocity of real drummers.**

Most humanisers add random jitter. wobblemidi instead samples per-instrument,
per-grid-position deviation distributions learned from the
[Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) — 341 rock
performances (114,890 hits) recorded by real drummers on a Roland TD-11 — and
drives them with a drifting kit-wide clock so the kit breathes together instead
of scattering. Ghost notes stay ghosts, backbeats stay solid, flams stay flams.
The deviations sound human because they are.

## Hear it

Every pair below is one run of `wobblemidi` at default settings on a pattern
programmed dead on the grid — same kit, same mix, only the MIDI differs.

<!-- AUDIO: uncomment each row as renders land in demo/audio/ (see demo/README.md)
| Pattern | Programmed | Humanised |
|---|---|---|
| Ghost-note rock beat (busy 16th hats) | [before](demo/audio/rock_ghosts_before.mp3) | [after](demo/audio/rock_ghosts_after.mp3) |
| Four-on-the-floor | [before](demo/audio/four_floor_before.mp3) | [after](demo/audio/four_floor_after.mp3) |
-->

*Audio renders are on their way. Until then:* the seeded, reproducible MIDI
pairs are in [`demo/`](demo/) — drag `rock_ghosts_input.mid` and
`rock_ghosts_humanised.mid` onto the same drum kit and A/B them.
[`demo/README.md`](demo/README.md) says what to listen for in each pair.

## Quickstart

```bash
pip install wobblemidi   # or: pip install git+https://github.com/JakebGutierrez/wobblemidi
wobblemidi drums.mid drums_humanised.mid
```

That's it — rock profile, intensity 0.35. Turn it up for a looser feel:

```bash
wobblemidi drums.mid drums_humanised.mid --intensity 0.5
```

There's also a GUI (drag a file in, turn knobs, audition, export):

```bash
pip install "wobblemidi[gui]"
wobblemidi-gui drums.mid
```

macOS users can grab the prebuilt app from
[Releases](https://github.com/JakebGutierrez/wobblemidi/releases) instead.

## What's in v1

- **Statistical timing & velocity** — offsets and velocity deltas sampled (via
  per-bucket KDE) from distributions keyed on instrument, grid position,
  beat/fill context, and dynamic tier. Not noise: the actual shape of how
  drummers deviate, per instrument, per position in the bar.
- **One clock for the kit** (`--groove-tightness`) — an AR(1) drift replaces
  independent per-hit jitter: hits wander *together* as a pocket, and the knob
  trades twitchy-vs-pocketed at a constant amount of spread. Kick velocity gets
  its own drift clock (the one instrument where real drummers' velocities are
  strongly autocorrelated — measured, not assumed).
- **Flam & chord preservation** — simultaneous hits move as one, and close
  ornaments (flams, grace notes, anything within a 12 ms window) shift rigidly
  instead of scattering. See `demo/flam_beat_*.mid` for the A/B.
- **Relative velocity tiering** — ghost/accent roles are read from *your*
  file's velocity structure, so an all-ghost verse or a two-level hat part
  keeps its programmed dynamics instead of collapsing to the dataset average.
- **Timing centred on the grid by default** — the source drummers' systematic
  push/drag is removed (kept as spread, not lean); `--push` restores the
  authentic lean if you want it.
- **Controls** — `--intensity` (0.2–0.5 is the useful range), `--section
  beat|fill`, `--seed` for reproducible output, `--timing-only` /
  `--velocity-only`, `--all-channels`. Per-lane intensity and a lean amount
  are available in the GUI and Python API.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--intensity` | `0.35` | Humanisation strength, 0.0–1.0. 0.2–0.5 is the useful range; higher values reproduce the raw drummer spread from the source recordings and will sound loose. |
| `--groove-tightness` | `0.4` | 0.0–<1.0. How much the kit shares one drifting internal clock. `0` = every hit timed independently (twitchy); higher values make hits wander together as a pocket. The amount of spread stays roughly constant — the knob changes its *character*. |
| `--section` | `beat` | Pass `fill` for fill passages — uses a separate timing distribution. |
| `--genre` | `rock` | Profile to use. Only `rock` ships in v1. |
| `--seed` | none | Integer seed for byte-reproducible output. |
| `--push` | off | Include the directional timing tendencies of the source drummers. Without it, variation is centred on the grid. |
| `--timing-only` / `--velocity-only` | off | Apply one axis of humanisation only (mutually exclusive). |
| `--all-channels` | off | Humanise drum-range notes on every MIDI channel, not just channel 10. |
| `--verbose` | off | Print the profile-fallback level used for each hit. |

## How it works

The profile build snaps every hit in the dataset to its grid position and
records `(timing offset, velocity residual)` pairs per bucket —
`genre|section|instrument|tier|grid position`, with a six-level fallback chain
for sparse buckets and per-bucket outlier clipping. Velocity deltas are
*residuals* against a per-take, per-position baseline, not raw deviations from
a global average — that distinction is what keeps your programmed accent
structure intact (the naive version audibly destroys it: soft hats slamming to
127). At apply time, a fitted KDE per bucket generates deviations, an AR(1)
clock correlates them across the kit (phi calibrated at ~0.37 from the data),
and a coupling window keeps chords and ornaments rigid. Full design docs:
[`wobblemidi_rebuild_spec.md`](wobblemidi_rebuild_spec.md) and its
[addendum](wobblemidi_rebuild_spec_addendum.md).

**Validation is measured, not vibes.** Profile changes gate on a harness
([`scripts/validate.py`](scripts/validate.py)) that scores the engine against
*held-out* human takes (GMD's own train/test split — the profile never sees
the takes it's graded on): per-instrument offset distributions
(Wasserstein/KS/σ/mean), within-position velocity spread, adjacent-jump
distributions, contour preservation, lag-1 autocorrelation, cross-instrument
gap tightness — each scored two-sided against human, so "more robotic than a
drummer" and "sloppier than a drummer" both fail. And then ear-tested, because
the metrics once passed a default that sounded wrong.

Two known misses are accepted and documented rather than hidden: at full
intensity, kick/hat/ride velocities carry 2–3× a human's within-role spread
(masked at the 0.35 default; the next lever is identified), and the snare
emits slightly too few *exactly repeated* velocities compared with a human
ghost run (a structural property of continuous KDE sampling — cosmetic).

## Engineering

This repo doubles as the **reference implementation** for a planned port, so
its behaviour is contract-locked:

- **Determinism** — one seed, three isolated RNG streams (samples, residuals,
  velocity clock), so e.g. two renders at different `--groove-tightness` draw
  identical samples and A/B only the clock. Seed semantics, draw order, and
  rounding rules are frozen in
  [`wobblemidi_determinism.md`](wobblemidi_determinism.md).
- **Golden vectors** — 26 byte-locked outputs over 10 input fixtures pin the
  entire parameter surface; they run in every pytest/CI pass (308 tests) and
  are byte-verified cross-platform (Linux + macOS CI matrix). Any behaviour
  change is a red diff, never an accident.
- **Porting contract** — the endgame is a JUCE/C++ Audio Unit plugin for Logic
  (offline whole-clip processing, Superior Drummer-style drag-in/drag-out).
  [`wobblemidi_porting_contract.md`](wobblemidi_porting_contract.md) defines
  the two-tier correctness gate a port must pass against this engine;
  [`wobblemidi_streamability.md`](wobblemidi_streamability.md) is the
  behaviour inventory proving the offline-clip model.

## MIDI compatibility

- Type 0 and type 1 MIDI files (type 2 is rejected)
- Roland TD-11 note mapping (GM-compatible, plus notes 22/26 hi-hat edge variants)
- Channel 10 only by default (`--all-channels` opts out)
- 4/4, 3/4, and other straight meters on a 16th grid; 6/8 auto-detected and
  handled on an 8th grid (grid-position awareness applies to 4/4 only)

## Building profiles from GMD

The bundled `rock.json` ships with the package. To rebuild it, download the
[Groove MIDI Dataset v1.0.0](https://magenta.tensorflow.org/datasets/groove)
and run:

```bash
python scripts/build_profiles.py /path/to/groove-v1.0.0/
```

Rebuilds gate on `scripts/validate.py` against held-out data, then an ear test.
Jazz and funk are deliberately out of scope in v1 — swing feel is
misrepresented on a straight 16th grid, and shipping a bad jazz profile is
worse than shipping none.

## Roadmap

No dates, no promises:

- **Logic plugin** — JUCE/AU, offline drag-in/process/drag-out; this Python
  engine is the reference it will be validated against.
- **Preserve-intent mode** — humanise *on top of* an already-played part
  instead of flatten-then-regenerate.
- More genres, when they can be done honestly (swing needs a different grid).

## License

[MIT](LICENSE). Profile data is derived from the Groove MIDI Dataset,
© Google LLC, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
