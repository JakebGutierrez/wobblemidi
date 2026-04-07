# pocketmidi

A CLI tool that humanises programmed drum MIDI files using real drummer performance data.

## Why

Most humanisation tools apply random noise or hand-coded guesswork. pocketmidi samples from real drummer timing and velocity data instead — so the deviations sound human because they are.

## Usage
```bash
pocketmidi drums.mid drums_human.mid --genre rock --intensity 0.7 --seed 42 --section beat
```

`--intensity` scales how much humanisation is applied (0.0 = none, 1.0 = full).  
`--section` declares whether the passage is a beat or fill (default: beat).  
`--seed` makes output reproducible.

## Status

v1 in development. Rock genre only. Straight 16th note grid.

## Stack

Python, mido, numpy, scipy, pandas, click

## Data

Uses the [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) (Magenta/Google).
Download and unzip it, then run:
```bash
python scripts/build_profiles.py /path/to/groove-v1.0.0/
```
