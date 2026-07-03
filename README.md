# pocketmidi

A CLI tool that humanises programmed drum MIDI using real drummer timing and velocity data.

Most humanisation tools apply random noise or hand-coded guesswork. pocketmidi samples from actual drummer performances instead — so the deviations sound human because they are.

## Install

```bash
pip install pocketmidi
```

Or from source:

```bash
git clone https://github.com/JakebGutierrez/pocketmidi
cd pocketmidi
pip install -e ".[dev]"
```

## Quick start

```bash
pocketmidi drums.mid drums_humanised.mid
```

By default this applies humanisation at `--intensity 0.35` using the rock profile. Turn it up
for a looser feel:

```bash
pocketmidi drums.mid drums_humanised.mid --intensity 0.5
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--intensity` | `0.35` | Humanisation strength, 0.0–1.0. 0.2–0.5 is the useful range; higher values reproduce the raw drummer spread from the source recordings and will sound loose. |
| `--section` | `beat` | Pass `fill` for fill passages — uses a separate timing distribution. |
| `--genre` | `rock` | Profile to use. Only `rock` is included in v1. |
| `--seed` | none | Integer seed for reproducible output. |
| `--groove-tightness` | `0.4` | 0.0–<1.0. How much the kit shares one drifting internal clock. `0` = every hit timed independently (twitchy); higher values make hits wander together as a pocket, and land simultaneously-notated hits (kick+snare) together. The overall amount of timing spread stays roughly the same — the knob mainly changes its *character* (how correlated hits are). |
| `--all-channels` | off | Humanise drum-range notes on every MIDI channel. By default only channel 10 (the standard drum channel) is touched, so melodic parts that happen to use drum-range note numbers are left alone. |
| `--timing-only` | off | Apply timing humanisation only; leave velocities unchanged. |
| `--velocity-only` | off | Apply velocity humanisation only; leave timing unchanged. |
| `--push` | off | Include the directional timing tendencies of the source drummers. Without this flag, timing variation is centred on the grid — natural human imprecision without systematic push or drag. Use `--push` if you want a specific "leaning into the beat" feel that matches the original recordings. |
| `--verbose` | off | Print the fallback level used for each hit. |

`--timing-only` and `--velocity-only` are mutually exclusive.

## How it works

pocketmidi builds a statistical profile from the [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) — a collection of real drummer performances recorded on a Roland TD-11 electronic kit. For each instrument (kick, snare, hi-hat, etc.) and grid position, it captures the distribution of timing offsets and velocity deviations that real drummers produce.

When humanising, each note is snapped to the nearest 16th-note grid position, then a timing offset and velocity delta are sampled from the matching distribution and applied. The result is timing variation that reflects how an actual drummer plays, not a random number generator.

**Timing is centred on the grid by default.** The raw GMD data contains directional tendencies (some drummers consistently push certain beats ahead of the grid). Without `--push`, these are removed — you get the spread and feel of real drumming without inheriting a specific drummer's rushing or dragging habit.

**One clock for the kit (`--groove-tightness`).** Sampling every hit's timing independently sounds twitchy — real drummers have a single internal clock that drifts slowly. pocketmidi keeps one running timing drift for the whole kit — even when kick, snare, and hats live on separate tracks: each hit nudges it, and the hit is placed by that shared drift plus a small independent wiggle, so the kit breathes together instead of scattering. The overall amount of timing variation stays about the same at any setting; the knob mainly changes how correlated it is. Hits notated on the same tick share one nudge, across tracks too, so kit-wide accents stay tight instead of flamming.

## MIDI compatibility

- Type 0 and type 1 MIDI files only
- Roland TD-11 note mapping (GM-compatible plus notes 22 and 26 for hi-hat edge variants)
- Only MIDI channel 10 (the standard drum channel) is humanised by default — pass `--all-channels` to include drum-range notes on other channels
- 4/4, 3/4, and other straight-grid time signatures (grid-position awareness applies to 4/4 only; other meters use the per-instrument distributions)
- 6/8 files are supported — uses an eighth-note grid automatically

## Building profiles from GMD

The bundled `rock.json` profile is pre-built and ships with the package. If you want to rebuild it (e.g. after modifying `build_profiles.py`), download the [Groove MIDI Dataset v1.0.0](https://magenta.tensorflow.org/datasets/groove) and run:

```bash
python scripts/build_profiles.py /path/to/groove-v1.0.0/
```

Profile data is derived from the Groove MIDI Dataset, which is © Google LLC and licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Status

Rock genre only. Straight 16th-note grid (8th-note for 6/8). Jazz and funk are out of scope in v1 — swing feel is not well-represented under a straight grid.
