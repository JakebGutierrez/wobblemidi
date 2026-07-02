"""Regression tests for the pre-refactor review fixes.

1. Simultaneous note_ons (chords) must not be force-separated by EPSILON_TICKS.
2. KDE samples must be clamped to each bucket's clipped data range.
"""
from __future__ import annotations

import numpy as np
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from importlib.resources import as_file, files

from pocketmidi.humanise import load_profile, humanise, _sample_bucket


def _note_ons(path):
    m = mido.MidiFile(path)
    t = 0
    out = []
    for msg in m.tracks[0]:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            out.append((t, msg.note))
    return out


def _chord_midi(path, ppq=480):
    mid = MidiFile(type=1, ticks_per_beat=ppq)
    tr = MidiTrack()
    mid.tracks.append(tr)
    tr.append(MetaMessage("set_tempo", tempo=500000, time=0))
    tr.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    raw = []
    for bar in range(2):
        base = bar * 16 * (ppq // 4)
        for beat in (4, 12):  # backbeats
            t = base + beat * (ppq // 4)
            raw.append((t, 36, 110))  # kick
            raw.append((t, 38, 105))  # snare — exactly simultaneous
    ev = []
    for t, n, v in raw:
        ev.append((t, "on", n, v))
        ev.append((t + 10, "off", n, 0))
    ev.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))
    prev = 0
    for t, kind, n, v in ev:
        # channel=9: the drum channel — required since the drum-channel filter,
        # otherwise these notes pass through and the test exercises nothing.
        tr.append(Message("note_on", channel=9, note=n,
                          velocity=v if kind == "on" else 0, time=t - prev))
        prev = t
    mid.save(path)


def _load_rock():
    with as_file(files("pocketmidi.profiles").joinpath("rock.json")) as p:
        return load_profile(p)


def test_simultaneous_hits_stay_on_same_tick_at_zero_intensity(tmp_path):
    src = tmp_path / "chord.mid"
    out = tmp_path / "chord_out.mid"
    _chord_midi(str(src))
    humanise(str(src), str(out), _load_rock(), intensity=0.0, seed=1)

    ons = _note_ons(str(out))
    # Group by tick: every backbeat should have BOTH kick(36) and snare(38)
    from collections import defaultdict
    by_tick = defaultdict(set)
    for t, n in ons:
        by_tick[t].add(n)
    chords = [notes for notes in by_tick.values() if 36 in notes or 38 in notes]
    assert len(chords) == 4, f"expected 4 backbeats, got {sorted(by_tick)}"
    for notes in chords:
        assert notes == {36, 38}, f"chord split across ticks: {by_tick}"


def test_kde_samples_clamped_to_bucket_range():
    prof = _load_rock()
    np.random.seed(0)
    # snare|hard|4 is a well-populated, KDE-backed bucket
    bucket = prof.buckets["rock|beat|snare|hard|4"]
    off_lo, off_hi = bucket.offsets.min(), bucket.offsets.max()
    vel_lo, vel_hi = bucket.vel_deltas.min(), bucket.vel_deltas.max()
    for _ in range(3000):
        off, vel = _sample_bucket(bucket)
        assert off_lo <= off <= off_hi, f"offset {off} escaped [{off_lo}, {off_hi}]"
        assert vel_lo <= vel <= vel_hi, f"vel_delta {vel} escaped [{vel_lo}, {vel_hi}]"
