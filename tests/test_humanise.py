"""Tests for pocketmidi/humanise.py"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import mido
import numpy as np
import pytest

from pocketmidi.humanise import (
    COUPLED_RESIDUAL_MS,
    EPSILON_TICKS,
    RESIDUAL_SHARE,
    BucketProfile,
    GrooveDrift,
    LoadedProfile,
    _lookup,
    _ms_offset_to_ticks,
    _velocity_tier,
    humanise,
    load_profile,
)
from pocketmidi.midi_utils import build_tempo_map, ticks_to_ms_with_map

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_PPQ = 480
DEFAULT_TEMPO_US = 500_000  # 120 BPM


def _profile_dict(**extra) -> dict:
    """Minimal profile with one rock|beat|kick bucket and optional extras."""
    base = {"rock|beat|kick": [[0.0, 0.0], [5.0, 10.0], [-5.0, -10.0]]}
    base.update(extra)
    return base


def _write_profile(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "rock.json"
    p.write_text(json.dumps(data))
    return p


def _make_midi(
    messages: list[mido.Message],
    ppq: int = DEFAULT_PPQ,
    tempo_us: int = DEFAULT_TEMPO_US,
    midi_type: int = 0,
) -> mido.MidiFile:
    """Build a single-track MidiFile from a list of messages (delta-time based)."""
    mid = mido.MidiFile(type=midi_type, ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    for msg in messages:
        track.append(msg)
    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def _save_load(mid: mido.MidiFile, tmp_path: Path) -> Path:
    p = tmp_path / "input.mid"
    mid.save(str(p))
    return p


def _collect_abs(mid: mido.MidiFile) -> list[tuple[int, mido.Message]]:
    """Return (abs_tick, msg) for all messages in track 0, filtering end_of_track."""
    result = []
    abs_tick = 0
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        if not isinstance(msg, mido.MetaMessage) or msg.type != "end_of_track":
            result.append((abs_tick, msg))
    return result


# ---------------------------------------------------------------------------
# TestLoadProfile
# ---------------------------------------------------------------------------

class TestLoadProfile:
    def test_loads_arrays(self, tmp_path):
        prof_path = _write_profile(tmp_path, {"rock|beat|kick": [[1.0, 2.0], [3.0, 4.0]]})
        profile = load_profile(prof_path)
        assert "rock|beat|kick" in profile.buckets
        bucket = profile.buckets["rock|beat|kick"]
        np.testing.assert_array_almost_equal(bucket.offsets, [1.0, 3.0])
        np.testing.assert_array_almost_equal(bucket.vel_deltas, [2.0, 4.0])

    def test_array_shape(self, tmp_path):
        prof_path = _write_profile(tmp_path, {"rock|beat|snare": [[0.0, 5.0]]})
        profile = load_profile(prof_path)
        bucket = profile.buckets["rock|beat|snare"]
        assert bucket.offsets.shape == (1,)
        assert bucket.vel_deltas.shape == (1,)

    def test_empty_bucket_skipped(self, tmp_path):
        prof_path = _write_profile(tmp_path, {"rock|beat|kick": [], "rock|beat|snare": [[1.0, 2.0]]})
        profile = load_profile(prof_path)
        assert "rock|beat|kick" not in profile.buckets
        assert "rock|beat|snare" in profile.buckets

    def test_kde_fitted(self, tmp_path):
        # 3 non-identical points → KDE fits successfully; kde.d must be 2 (2D KDE).
        from scipy.stats import gaussian_kde
        prof_path = _write_profile(tmp_path, {"rock|beat|kick": [[0.0, 0.0], [5.0, 3.0], [-5.0, 8.0]]})
        profile = load_profile(prof_path)
        bucket = profile.buckets["rock|beat|kick"]
        assert isinstance(bucket.kde, gaussian_kde)
        assert bucket.kde.d == 2

    def test_kde_none_for_degenerate_bucket(self, tmp_path):
        # 1 sample → scipy raises ValueError → kde falls back to None.
        prof_path = _write_profile(tmp_path, {"rock|beat|kick": [[3.0, 5.0]]})
        profile = load_profile(prof_path)
        bucket = profile.buckets["rock|beat|kick"]
        assert bucket.kde is None

    def test_meta_velocity_thresholds(self, tmp_path):
        data = {
            "_meta": {"velocity_thresholds": {"snare": [40.0, 85.0]}, "kde_bw_method": "scott"},
            "rock|beat|snare": [[0.0, 0.0], [5.0, 5.0], [-5.0, -5.0]],
        }
        prof_path = _write_profile(tmp_path, data)
        profile = load_profile(prof_path)
        assert "snare" in profile.velocity_thresholds
        low, high = profile.velocity_thresholds["snare"]
        assert low == pytest.approx(40.0)
        assert high == pytest.approx(85.0)
        # _meta must not appear as a bucket
        assert "_meta" not in profile.buckets

    @pytest.mark.parametrize("bad_bw", [
        "not_a_real_method",   # invalid string
        {"key": "val"},        # object — would reach gaussian_kde and be swallowed
        [1, 2],                # array — same problem
    ])
    def test_invalid_bw_method_raises(self, tmp_path, bad_bw):
        # Any invalid kde_bw_method in _meta must raise ValueError immediately,
        # not silently set kde=None for every bucket (broken _meta contract).
        data = {
            "_meta": {"kde_bw_method": bad_bw},
            "rock|beat|kick": [[0.0, 0.0], [5.0, 10.0], [-5.0, -10.0]],
        }
        prof_path = _write_profile(tmp_path, data)
        with pytest.raises(ValueError, match="kde_bw_method"):
            load_profile(prof_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_profile(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# TestLookup
# ---------------------------------------------------------------------------

def _make_bucket(*pairs) -> BucketProfile:
    """Build a minimal BucketProfile from (offset_ms, vel_delta) pairs (kde=None)."""
    arr = np.array(pairs, dtype=float)
    return BucketProfile(offsets=arr[:, 0], vel_deltas=arr[:, 1], kde=None)


class TestLookup:
    def setup_method(self):
        self.profile = LoadedProfile(
            buckets={
                "rock|beat|kick":       _make_bucket((1.0, 0.0)),
                "rock|beat|snare":      _make_bucket((2.0, 0.0)),
                "global|hihat_closed":  _make_bucket((0.5, 0.0)),
            },
            velocity_thresholds={},
        )

    def test_level1_exact(self):
        bucket, level, _ = _lookup(self.profile, "rock", "beat", "kick", 80)
        assert level == 1
        assert bucket is not None

    def test_level2_beat_fallback(self):
        # fill context not in buckets → should fall to beat (unstratified)
        bucket, level, _ = _lookup(self.profile, "rock", "fill", "snare", 80)
        assert level == 2
        assert bucket is not None

    def test_level3_global(self):
        # hi-hat: 3-level chain; fill not present → beat not present → global
        bucket, level, _ = _lookup(self.profile, "rock", "fill", "hihat_closed", 80)
        assert level == 3
        assert bucket is not None

    def test_total_miss(self):
        bucket, level, _ = _lookup(self.profile, "rock", "beat", "ride", 80)
        assert bucket is None
        assert level is None

    def test_tier_routing_soft(self):
        # Snare with velocity thresholds: soft tier key present → level 1.
        profile = LoadedProfile(
            buckets={
                "rock|beat|snare|soft":   _make_bucket((1.0, 0.0), (2.0, 1.0), (-1.0, -1.0)),
                "rock|beat|snare":        _make_bucket((0.0, 0.0)),
            },
            velocity_thresholds={"snare": (40.0, 80.0)},
        )
        bucket, level, _ = _lookup(profile, "rock", "beat", "snare", 30)  # 30 < 40 → soft
        assert level == 1
        assert bucket is not None

    def test_tier_routing_medium(self):
        profile = LoadedProfile(
            buckets={
                "rock|beat|snare|medium": _make_bucket((1.0, 0.0), (2.0, 1.0), (-1.0, -1.0)),
                "rock|beat|snare":        _make_bucket((0.0, 0.0)),
            },
            velocity_thresholds={"snare": (40.0, 80.0)},
        )
        bucket, level, _ = _lookup(profile, "rock", "beat", "snare", 60)  # 40 <= 60 < 80 → medium
        assert level == 1
        assert bucket is not None

    def test_tier_routing_hard(self):
        profile = LoadedProfile(
            buckets={
                "rock|beat|snare|hard":   _make_bucket((1.0, 0.0), (2.0, 1.0), (-1.0, -1.0)),
                "rock|beat|snare":        _make_bucket((0.0, 0.0)),
            },
            velocity_thresholds={"snare": (40.0, 80.0)},
        )
        bucket, level, _ = _lookup(profile, "rock", "beat", "snare", 100)  # 100 >= 80 → hard
        assert level == 1
        assert bucket is not None

    def test_tier_drop_fallback(self):
        # Exact tier key absent → falls back to unstratified at level 2.
        profile = LoadedProfile(
            buckets={
                "rock|beat|snare": _make_bucket((0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)),
            },
            velocity_thresholds={"snare": (40.0, 80.0)},
        )
        bucket, level, _ = _lookup(profile, "rock", "beat", "snare", 30)  # soft key absent
        assert level == 2
        assert bucket is not None

    def test_hihat_no_tier_routing(self):
        # Hi-hat has no velocity thresholds → goes straight to the 3-level chain.
        # Velocity value must not affect key selection.
        profile = LoadedProfile(
            buckets={
                "rock|beat|hihat_closed": _make_bucket((0.5, 0.0), (1.0, 0.5), (-0.5, -0.5)),
            },
            velocity_thresholds={"snare": (40.0, 80.0)},  # snare thresholds present, hihat absent
        )
        bucket_soft, level_soft, _ = _lookup(profile, "rock", "beat", "hihat_closed", 20)
        bucket_hard, level_hard, _ = _lookup(profile, "rock", "beat", "hihat_closed", 120)
        assert level_soft == 1
        assert level_hard == 1  # velocity has no effect; same bucket


# ---------------------------------------------------------------------------
# TestMsOffsetToTicks
# ---------------------------------------------------------------------------

class TestMsOffsetToTicks:
    def _simple_map(self):
        return [(0, DEFAULT_TEMPO_US)]  # 120 BPM

    def test_zero(self):
        assert _ms_offset_to_ticks(0, 0.0, self._simple_map(), DEFAULT_PPQ) == 0

    def test_positive(self):
        # At 120 BPM, 1 beat = 500ms, PPQ=480 → 1 tick ≈ 500000/480/1000 ms ≈ 1.0417 ms
        # 10ms → ~9.6 ticks → rounds to 10
        result = _ms_offset_to_ticks(0, 10.0, self._simple_map(), DEFAULT_PPQ)
        ms_per_tick = DEFAULT_TEMPO_US / DEFAULT_PPQ / 1000.0
        expected = round(10.0 / ms_per_tick)
        assert result == expected

    def test_negative(self):
        grid_tick = DEFAULT_PPQ  # one beat in
        result = _ms_offset_to_ticks(grid_tick, -10.0, self._simple_map(), DEFAULT_PPQ)
        ms_per_tick = DEFAULT_TEMPO_US / DEFAULT_PPQ / 1000.0
        expected = -round(10.0 / ms_per_tick)
        assert result == expected

    def test_positive_across_tempo_boundary(self):
        # Tempo map: 120 BPM from tick 0, 60 BPM from tick 480
        # grid_tick=0, want to walk 10ms forward crossing the boundary at tick 480
        # At 120 BPM: ms_per_tick = 500000/480/1000 ≈ 1.0417ms → tick 480 is 500ms away
        # So 10ms fits entirely in the first segment
        tempo_map = [(0, DEFAULT_TEMPO_US), (480, 1_000_000)]
        result = _ms_offset_to_ticks(0, 10.0, tempo_map, DEFAULT_PPQ)
        assert result > 0  # moved forward

    def test_backward_starting_on_boundary(self):
        # Was an infinite-loop bug: current == tempo_map[idx][0] → ticks_to_prev == 0
        tempo_map = [(0, DEFAULT_TEMPO_US), (480, 1_000_000)]
        grid_tick = 480  # exactly on the boundary
        result = _ms_offset_to_ticks(grid_tick, -5.0, tempo_map, DEFAULT_PPQ)
        assert result < 0  # moved backward

    def test_across_forward_tempo_boundary_crosses(self):
        # Tempo map: very slow first segment so 10ms spans into the second segment
        # 240000 us/beat at PPQ=480 → ms_per_tick = 240000/480/1000 = 0.5ms
        # tick 0→4: 4 ticks * 0.5ms = 2ms, then switch to 500000us
        tempo_map = [(0, 240_000), (4, DEFAULT_TEMPO_US)]
        # walk 10ms from tick 0: first 2ms uses 4 ticks, remaining 8ms at 120BPM
        result = _ms_offset_to_ticks(0, 10.0, tempo_map, DEFAULT_PPQ)
        ms_per_tick_after = DEFAULT_TEMPO_US / DEFAULT_PPQ / 1000.0
        expected = 4 + round(8.0 / ms_per_tick_after)
        assert result == expected


# ---------------------------------------------------------------------------
# Shared fixture: build a mid + profile + output path
# ---------------------------------------------------------------------------

def _run_humanise(mid, profiles_dict, tmp_path, **kwargs):
    inp = tmp_path / "in.mid"
    out = tmp_path / "out.mid"
    mid.save(str(inp))
    prof_path = _write_profile(tmp_path, profiles_dict)
    profiles = load_profile(prof_path)
    humanise(inp, out, profiles, **kwargs)
    return mido.MidiFile(str(out))


# ---------------------------------------------------------------------------
# TestHumaniseVelocityClamp
# ---------------------------------------------------------------------------

class TestHumaniseVelocityClamp:
    def test_clamp_below_1(self, tmp_path):
        # kick at velocity 1 with large negative delta → should clamp to 1
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=1,   time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0,   time=480),
        ])
        # vel_delta always -100
        profiles = {"rock|beat|kick": [[-0.0, -100.0]]}
        out = _run_humanise(mid, profiles, tmp_path, genre="rock", beat_type="beat", seed=0)
        note_ons = [
            msg for _, msg in _collect_abs(out)
            if msg.type == "note_on" and msg.velocity > 0
        ]
        assert note_ons[0].velocity >= 1

    def test_clamp_above_127(self, tmp_path):
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=127, time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0,   time=480),
        ])
        profiles = {"rock|beat|kick": [[0.0, 100.0]]}
        out = _run_humanise(mid, profiles, tmp_path, genre="rock", beat_type="beat", seed=0)
        note_ons = [
            msg for _, msg in _collect_abs(out)
            if msg.type == "note_on" and msg.velocity > 0
        ]
        assert note_ons[0].velocity <= 127


# ---------------------------------------------------------------------------
# TestHumaniseNoDeltaTimeNegative
# ---------------------------------------------------------------------------

class TestHumaniseNoDeltaTimeNegative:
    def _check_no_negative_deltas(self, out_mid):
        for track in out_mid.tracks:
            for msg in track:
                assert msg.time >= 0, f"Negative delta: {msg}"

    def test_no_negative_deltas_basic(self, tmp_path):
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=80, time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
            mido.Message("note_on",  channel=9, note=36, velocity=80, time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
        ])
        profiles = {"rock|beat|kick": [[-20.0, 0.0], [20.0, 0.0]]}
        out = _run_humanise(mid, profiles, tmp_path, genre="rock", beat_type="beat", seed=1)
        self._check_no_negative_deltas(out)

    def test_shifted_early_bounded_by_prev_emitted(self, tmp_path):
        # note at tick 480 shifted early must land >= 0
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=80, time=480),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
        ])
        profiles = {"rock|beat|kick": [[-10000.0, 0.0]]}  # huge early offset
        out = _run_humanise(mid, profiles, tmp_path, genre="rock", beat_type="beat", seed=0)
        self._check_no_negative_deltas(out)

    def test_shifted_late_bounded_by_note_off(self, tmp_path):
        # note_on at 0, note_off at tick 10 → can't shift past tick 10
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=80, time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=10),
        ])
        profiles = {"rock|beat|kick": [[10000.0, 0.0]]}  # huge late offset
        out = _run_humanise(mid, profiles, tmp_path, genre="rock", beat_type="beat", seed=0)
        events = _collect_abs(out)
        note_on_abs = next(t for t, m in events if m.type == "note_on" and m.velocity > 0)
        note_off_abs = next(t for t, m in events if m.type == "note_off" or (m.type == "note_on" and m.velocity == 0))
        assert note_on_abs < note_off_abs

    def test_shifted_late_bounded_by_next_fixed(self, tmp_path):
        # note_on at 0, set_tempo at tick 240 → can't shift past 240
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=80, time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
        ])
        # Insert a CC (fixed event) at tick 100 — use separate track approach via direct track building
        mid2 = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid2.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=480))
        track.append(mido.Message("control_change", channel=9, control=7, value=100, time=-470))  # at tick 10
        track.append(mido.MetaMessage("end_of_track", time=0))
        # build simpler: note_on at 0, CC at 10, note_off at 480
        mid3 = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track3 = mido.MidiTrack()
        mid3.tracks.append(track3)
        track3.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track3.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))
        track3.append(mido.Message("control_change", channel=9, control=7, value=100, time=10))  # abs=10
        track3.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=470))  # abs=480
        track3.append(mido.MetaMessage("end_of_track", time=0))

        profiles = {"rock|beat|kick": [[10000.0, 0.0]]}
        inp = tmp_path / "in3.mid"
        out = tmp_path / "out3.mid"
        mid3.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", seed=0)
        result = mido.MidiFile(str(out))
        self._check_no_negative_deltas(result)
        # shifted note_on must be before the CC at abs=10
        events = _collect_abs(result)
        note_on_abs = next(t for t, m in events if m.type == "note_on" and m.velocity > 0)
        cc_abs = next(t for t, m in events if m.type == "control_change")
        assert note_on_abs < cc_abs


# ---------------------------------------------------------------------------
# TestHumaniseFixedEventsUnmoved
# ---------------------------------------------------------------------------

class TestHumaniseFixedEventsUnmoved:
    def test_cc_stays_at_original_tick(self, tmp_path):
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=480))
        track.append(mido.Message("control_change", channel=0, control=7, value=100, time=0))  # abs=480
        track.append(mido.MetaMessage("end_of_track", time=0))

        profiles = {"rock|beat|kick": [[5.0, 0.0]]}
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", seed=0)
        result = mido.MidiFile(str(out))

        events = _collect_abs(result)
        cc_abs = next(t for t, m in events if m.type == "control_change")
        assert cc_abs == 480


# ---------------------------------------------------------------------------
# TestHumaniseEpsilonDropped
# ---------------------------------------------------------------------------

class TestHumaniseEpsilonDropped:
    def test_epsilon_dropped_when_window_exhausted(self, tmp_path):
        # Two consecutive hits at same tick; shift late. The second hit's
        # lower_with_eps could exceed upper — epsilon must be dropped.
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        # First kick at tick 0, note_off at tick 1
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=1))
        # Second kick at tick 1, note_off at tick 2
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=1))
        track.append(mido.MetaMessage("end_of_track", time=0))

        # Profile: big late offset so both hits try to land at tick 1; second is
        # bounded above by its note_off at tick 2
        profiles = {"rock|beat|kick": [[10000.0, 0.0]]}
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", seed=0)
        result = mido.MidiFile(str(out))

        for track in result.tracks:
            for msg in track:
                assert msg.time >= 0

    def test_fixed_events_still_unmoved_after_epsilon_drop(self, tmp_path):
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=1))
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=1))
        track.append(mido.Message("control_change", channel=0, control=7, value=100, time=0))  # abs=2
        track.append(mido.MetaMessage("end_of_track", time=0))

        profiles = {"rock|beat|kick": [[10000.0, 0.0]]}
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", seed=0)
        result = mido.MidiFile(str(out))

        events = _collect_abs(result)
        cc_abs = next(t for t, m in events if m.type == "control_change")
        assert cc_abs == 2

    def test_same_tick_collision_empty_window(self, tmp_path):
        # Shiftable kick at tick 0, fixed CC also at tick 0 → upper_exclusive=0,
        # ceiling=-1, lower=0 > ceiling: empty window, note passes through at abs_t=0.
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))
        track.append(mido.Message("control_change", channel=0, control=7, value=100, time=0))  # abs=0
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=480))
        track.append(mido.MetaMessage("end_of_track", time=0))

        profiles = {"rock|beat|kick": [[10000.0, 5.0]]}
        inp = tmp_path / "in_sc.mid"
        out = tmp_path / "out_sc.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", seed=0)
        result = mido.MidiFile(str(out))

        for track in result.tracks:
            for msg in track:
                assert msg.time >= 0

        events = _collect_abs(result)
        kick_abs = next(t for t, m in events if m.type == "note_on" and m.velocity > 0)
        cc_abs = next(t for t, m in events if m.type == "control_change")
        assert kick_abs == 0       # stayed at original tick
        assert cc_abs == 0         # fixed event unmoved
        assert kick_abs < cc_abs or kick_abs == cc_abs  # no ordering violation (same-tick passthrough)


# ---------------------------------------------------------------------------
# TestHumaniseDenseHitAfterFixedNoteOn
# ---------------------------------------------------------------------------

class TestHumaniseDenseHitAfterFixedNoteOn:
    def test_shiftable_after_fixed_note_on(self, tmp_path):
        # non-drum note_on (no profile, channel 0) at tick 0 is fixed
        # drum kick at tick 0 with early offset should land >= 0 + EPSILON
        # unless bounded above
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.Message("note_on",  channel=0, note=60, velocity=64, time=0))   # fixed (no profile)
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))   # shiftable kick
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=480))
        track.append(mido.Message("note_off", channel=0, note=60, velocity=0,  time=0))
        track.append(mido.MetaMessage("end_of_track", time=0))

        profiles = {"rock|beat|kick": [[0.0, 0.0]]}  # zero offset → lands on grid
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", seed=0)
        result = mido.MidiFile(str(out))

        for track in result.tracks:
            for msg in track:
                assert msg.time >= 0


# ---------------------------------------------------------------------------
# TestHumaniseNoteOffStaysAfterNoteOn
# ---------------------------------------------------------------------------

class TestHumaniseNoteOffStaysAfterNoteOn:
    def test_note_off_after_note_on(self, tmp_path):
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=80, time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=5),
        ])
        profiles = {"rock|beat|kick": [[10000.0, 0.0]]}  # push late
        out = _run_humanise(mid, profiles, tmp_path, genre="rock", beat_type="beat", seed=0)
        events = _collect_abs(out)
        note_on_abs = next(t for t, m in events if m.type == "note_on" and m.velocity > 0)
        note_off_abs = next(t for t, m in events if m.type == "note_off" or (m.type == "note_on" and m.velocity == 0))
        assert note_on_abs < note_off_abs


# ---------------------------------------------------------------------------
# TestHumaniseMixedShiftableAndFixedSameNotePairing
# ---------------------------------------------------------------------------

class TestHumaniseMixedShiftableAndFixedSameNotePairing:
    def test_shiftable_bounded_by_its_own_note_off(self, tmp_path):
        # note_on (no-profile, ch0) at 0, note_off at 10
        # note_on (kick, ch9)   at 0, note_off at 20
        # Fixed kick: ch0/note60, shiftable: ch9/note36 (same note doesn't matter — different ch)
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.Message("note_on",  channel=0, note=60, velocity=64, time=0))
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=0, note=60, velocity=0,  time=10))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=10))
        track.append(mido.MetaMessage("end_of_track", time=0))

        profiles = {"rock|beat|kick": [[10000.0, 0.0]]}
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", seed=0)
        result = mido.MidiFile(str(out))

        for track in result.tracks:
            for msg in track:
                assert msg.time >= 0

        events = _collect_abs(result)
        kick_on = next(t for t, m in events if m.type == "note_on" and m.velocity > 0 and m.channel == 9)
        kick_off = next(t for t, m in events if (m.type == "note_off" or (m.type == "note_on" and m.velocity == 0)) and m.channel == 9)
        assert kick_on < kick_off


# ---------------------------------------------------------------------------
# TestHumaniseOverlappingSameNoteShiftablePairing
# ---------------------------------------------------------------------------

class TestHumaniseOverlappingSameNoteShiftablePairing:
    def test_fifo_pairing(self, tmp_path):
        # Two shiftable kicks: A at tick 0, B at tick 5
        # note_offs at tick 10 and tick 20 → A paired with tick 10, B with tick 20
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))   # A abs=0
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=5))   # B abs=5
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=5))   # note_off abs=10 → pairs with A
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=10))  # note_off abs=20 → pairs with B
        track.append(mido.MetaMessage("end_of_track", time=0))

        profiles = {"rock|beat|kick": [[10000.0, 0.0]]}
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", seed=0)
        result = mido.MidiFile(str(out))

        for track in result.tracks:
            for msg in track:
                assert msg.time >= 0


# ---------------------------------------------------------------------------
# TestHumaniseRejectsType2
# ---------------------------------------------------------------------------

class TestHumaniseType1MultiTrack:
    def test_type1_two_tracks_processed(self, tmp_path):
        # Type 1: tempo event lives in track 0 (conductor track), drum notes in track 1.
        # Verifies both tracks survive and the shared tempo map is applied correctly.
        mid = mido.MidiFile(type=1, ticks_per_beat=DEFAULT_PPQ)

        conductor = mido.MidiTrack()
        mid.tracks.append(conductor)
        conductor.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        conductor.append(mido.MetaMessage("end_of_track", time=0))

        drum_track = mido.MidiTrack()
        mid.tracks.append(drum_track)
        drum_track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=480))
        drum_track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=480))
        drum_track.append(mido.MetaMessage("end_of_track", time=0))

        profiles = {"rock|beat|kick": [[5.0, 5.0]]}
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", seed=0)

        result = mido.MidiFile(str(out))
        assert result.type == 1
        assert len(result.tracks) == 2

        for track in result.tracks:
            for msg in track:
                assert msg.time >= 0

    def test_type1_tempo_in_track1_applied_to_track0_drums(self, tmp_path):
        # Tempo event is in track 1; drum notes are in track 0.
        # Uses a non-default tempo (1_000_000 us = 60 BPM) so the expected output tick
        # differs from the 120-BPM fallback — the assertion fails if build_tempo_map()
        # only scanned track 0.
        NON_DEFAULT_TEMPO = 1_000_000  # 60 BPM: 1 tick ≈ 2.083 ms at PPQ=480
        mid = mido.MidiFile(type=1, ticks_per_beat=DEFAULT_PPQ)

        drum_track = mido.MidiTrack()
        mid.tracks.append(drum_track)
        drum_track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=480))
        drum_track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=480))
        drum_track.append(mido.MetaMessage("end_of_track", time=0))

        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=NON_DEFAULT_TEMPO, time=0))
        tempo_track.append(mido.MetaMessage("end_of_track", time=0))

        profiles = {"rock|beat|kick": [[10.0, 0.0]]}
        inp = tmp_path / "in2.mid"
        out = tmp_path / "out2.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", intensity=1.0, seed=0)

        result = mido.MidiFile(str(out))
        events = _collect_abs(result)
        note_on_abs = next(t for t, m in events if m.type == "note_on" and m.velocity > 0)

        ms_per_tick = NON_DEFAULT_TEMPO / DEFAULT_PPQ / 1000.0  # ≈ 2.083 ms
        expected = 480 + round(10.0 / ms_per_tick)              # ≈ 485 ticks
        assert note_on_abs == expected, (
            f"Expected {expected} ticks (60 BPM tempo from track 1); "
            f"got {note_on_abs} — tempo map may not have scanned track 1"
        )


class TestHumaniseRejectsType2:
    def test_raises_value_error(self, tmp_path):
        mid = mido.MidiFile(type=2, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("end_of_track", time=0))

        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        profiles = LoadedProfile(buckets={}, velocity_thresholds={})
        with pytest.raises(ValueError, match="Type 2"):
            humanise(inp, out, profiles)


# ---------------------------------------------------------------------------
# TestHumaniseSeedReproducible
# ---------------------------------------------------------------------------

class TestHumaniseSeedReproducible:
    def test_same_seed_same_output(self, tmp_path):
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=80, time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
            mido.Message("note_on",  channel=9, note=38, velocity=90, time=0),
            mido.Message("note_off", channel=9, note=38, velocity=0,  time=480),
        ])
        profiles_dict = {
            "rock|beat|kick":  [[-5.0, -5.0], [5.0, 5.0]],
            "rock|beat|snare": [[-3.0, 3.0],  [3.0, -3.0]],
        }

        inp = tmp_path / "in.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles_dict)
        profs = load_profile(prof_path)

        out1 = tmp_path / "out1.mid"
        out2 = tmp_path / "out2.mid"
        humanise(inp, out1, profs, genre="rock", beat_type="beat", seed=42)
        humanise(inp, out2, profs, genre="rock", beat_type="beat", seed=42)

        assert out1.read_bytes() == out2.read_bytes()

    def test_different_seed_may_differ(self, tmp_path):
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=80, time=480),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
        ])
        profiles_dict = {"rock|beat|kick": [[-5.0, -5.0], [5.0, 5.0], [0.0, 0.0]]}

        inp = tmp_path / "in.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles_dict)
        profs = load_profile(prof_path)

        out1 = tmp_path / "out1.mid"
        out2 = tmp_path / "out2.mid"
        humanise(inp, out1, profs, genre="rock", beat_type="beat", seed=1)
        humanise(inp, out2, profs, genre="rock", beat_type="beat", seed=2)
        # Not guaranteed to differ with only one hit, but with varied profile it usually will.
        # We test this by just verifying both complete without error.


# ---------------------------------------------------------------------------
# TestHumaniseUnknownNotes
# ---------------------------------------------------------------------------

class TestHumaniseUnknownNotes:
    def test_unknown_note_passes_through(self, tmp_path):
        # MIDI note 99 is not in TD11_TO_GROUP → must pass through unmodified
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=99, velocity=80, time=0),
            mido.Message("note_off", channel=9, note=99, velocity=0,  time=480),
        ])
        profiles = {"rock|beat|kick": [[5.0, 5.0]]}
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", seed=0)
        result = mido.MidiFile(str(out))

        events = _collect_abs(result)
        note_on = next(
            (t, m) for t, m in events if m.type == "note_on" and m.velocity > 0 and m.note == 99
        )
        assert note_on[1].velocity == 80  # velocity unchanged


# ---------------------------------------------------------------------------
# TestDrumChannelFilter
# ---------------------------------------------------------------------------

class TestDrumChannelFilter:
    """Only MIDI channel 10 (mido channel 9) is humanised by default.

    Melodic parts often use drum-range note numbers (36 = C2); without the
    channel filter they got "humanised" and corrupted.
    """

    # Constant bucket → deterministic: +50 ms ≈ +48 ticks at 120 BPM / PPQ 480,
    # vel_delta +10.
    _PROFILE = {"rock|beat|kick": [[50.0, 10.0], [50.0, 10.0], [50.0, 10.0]]}
    _SHIFT_TICKS = round(50.0 / (DEFAULT_TEMPO_US / DEFAULT_PPQ / 1000.0))  # 48

    def _two_channel_midi(self) -> mido.MidiFile:
        """Channel-0 melodic note on a drum-range number (36) + channel-9 kick."""
        return _make_midi([
            mido.Message("note_on",  channel=0, note=36, velocity=64, time=0),
            mido.Message("note_off", channel=0, note=36, velocity=0,  time=480),
            mido.Message("note_on",  channel=9, note=36, velocity=64, time=480),  # abs=960
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
        ])

    def _note_ons_by_channel(self, out: mido.MidiFile) -> dict[int, tuple[int, int]]:
        """Map channel → (abs_tick, velocity) of its first note_on."""
        result = {}
        for t, m in _collect_abs(out):
            if m.type == "note_on" and m.velocity > 0 and m.channel not in result:
                result[m.channel] = (t, m.velocity)
        return result

    def test_non_drum_channel_note_untouched_by_default(self, tmp_path):
        out = _run_humanise(self._two_channel_midi(), self._PROFILE, tmp_path, seed=0)
        ons = self._note_ons_by_channel(out)
        assert ons[0] == (0, 64), "channel-0 drum-range note must pass through untouched"
        assert ons[9] == (960 + self._SHIFT_TICKS, 74), "channel-9 kick must be humanised"

    def test_all_channels_reenables_other_channels(self, tmp_path):
        out = _run_humanise(self._two_channel_midi(), self._PROFILE, tmp_path,
                            seed=0, all_channels=True)
        ons = self._note_ons_by_channel(out)
        assert ons[0] == (0 + self._SHIFT_TICKS, 74), "channel-0 note must shift with all_channels"
        assert ons[9] == (960 + self._SHIFT_TICKS, 74)

    def test_defaults(self):
        import inspect
        params = inspect.signature(humanise).parameters
        assert params["all_channels"].default is False
        assert params["phi"].default == 0.4  # roadmap: phi default 0.5 → 0.4


# ---------------------------------------------------------------------------
# TestHumaniseIntensityZero
# ---------------------------------------------------------------------------

class TestHumaniseIntensityZero:
    def test_intensity_zero_lands_on_grid_with_unchanged_velocity(self, tmp_path):
        ppq = DEFAULT_PPQ
        # Put kick on an off-grid position: tick 10 → grid = 0 (nearest 16th at ppq=480 is 120)
        # Actually tick 10 snaps to 0. Use tick 130 → nearest 16th = 120.
        mid = mido.MidiFile(type=0, ticks_per_beat=ppq)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=130))  # snaps to 120
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=480))
        track.append(mido.MetaMessage("end_of_track", time=0))

        profiles = {"rock|beat|kick": [[20.0, 15.0]]}  # non-zero offset and vel_delta
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", intensity=0.0, seed=0)
        result = mido.MidiFile(str(out))

        events = _collect_abs(result)
        note_on_abs, note_on_msg = next(
            (t, m) for t, m in events if m.type == "note_on" and m.velocity > 0
        )
        assert note_on_abs == 120, f"Expected grid tick 120, got {note_on_abs}"
        assert note_on_msg.velocity == 80


# ---------------------------------------------------------------------------
# TestHumaniseNoProfile
# ---------------------------------------------------------------------------

class TestHumaniseNoProfile:
    def test_no_profile_note_passes_through(self, tmp_path):
        # Kick but profile dict is empty → passes through at original tick with original velocity
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=80, time=480),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
        ])
        profiles_dict = {}  # no profiles at all
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        prof_path = _write_profile(tmp_path, profiles_dict)
        profs = load_profile(prof_path)
        humanise(inp, out, profs, genre="rock", beat_type="beat", seed=0)
        result = mido.MidiFile(str(out))

        events = _collect_abs(result)
        note_on = next((t, m) for t, m in events if m.type == "note_on" and m.velocity > 0)
        assert note_on[0] == 480
        assert note_on[1].velocity == 80


# ---------------------------------------------------------------------------
# TestHumaniseFlags
# ---------------------------------------------------------------------------

_FLAGS_PROFILES = {"rock|beat|kick": [[5.0, 10.0]]}  # non-zero offset and vel_delta


class TestHumaniseFlags:
    def test_timing_only_preserves_velocity(self, tmp_path):
        # timing_only=True: offset applied, velocity untouched
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=64, time=480),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
        ])
        out = _run_humanise(mid, _FLAGS_PROFILES, tmp_path,
                            genre="rock", beat_type="beat", timing_only=True, seed=0)
        events = _collect_abs(out)
        note_ons = [(t, m) for t, m in events if m.type == "note_on" and m.velocity > 0]
        assert len(note_ons) == 1
        tick, msg = note_ons[0]
        assert msg.velocity == 64          # velocity unchanged
        assert tick != 480                 # timing was shifted

    def test_velocity_only_preserves_position(self, tmp_path):
        # velocity_only=True: velocity changed, position untouched (bypass path)
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=64, time=480),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
        ])
        out = _run_humanise(mid, _FLAGS_PROFILES, tmp_path,
                            genre="rock", beat_type="beat", velocity_only=True, seed=0)
        events = _collect_abs(out)
        note_ons = [(t, m) for t, m in events if m.type == "note_on" and m.velocity > 0]
        assert len(note_ons) == 1
        tick, msg = note_ons[0]
        assert tick == 480                 # position unchanged
        assert msg.velocity != 64          # velocity was shifted

    def test_velocity_only_same_tick_notes(self, tmp_path):
        # Two kicks on the exact same tick: both must stay at original position.
        # Regression anchor — the old `candidate = abs_t` approach would bump the
        # second note to abs_t + EPSILON_TICKS via the prev_note_on_abs lower bound.
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=64, time=480),
            mido.Message("note_on",  channel=9, note=36, velocity=64, time=0),   # same abs tick
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=0),
        ])
        out = _run_humanise(mid, _FLAGS_PROFILES, tmp_path,
                            genre="rock", beat_type="beat", velocity_only=True, seed=0)
        events = _collect_abs(out)
        note_on_ticks = [t for t, m in events if m.type == "note_on" and m.velocity > 0]
        assert len(note_on_ticks) == 2
        assert note_on_ticks[0] == 480     # first note at original tick
        assert note_on_ticks[1] == 480     # second note also at original tick (not bumped)

    def test_both_flags_raises(self, tmp_path):
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=64, time=480),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
        ])
        inp = _save_load(mid, tmp_path)
        out = tmp_path / "out.mid"
        prof_path = _write_profile(tmp_path, _FLAGS_PROFILES)
        profs = load_profile(prof_path)
        with pytest.raises(ValueError, match="mutually exclusive"):
            humanise(inp, out, profs, timing_only=True, velocity_only=True)


# ---------------------------------------------------------------------------
# TestLookupGridPos
# ---------------------------------------------------------------------------

class TestLookupGridPos:
    def test_stratified_exact_with_grid_pos(self):
        # Profile has the exact grid-pos + tier key → level 1.
        profile = LoadedProfile(
            buckets={
                "rock|beat|kick|hard|3": _make_bucket((1.0, 0.0), (2.0, 1.0), (-1.0, -1.0)),
                "rock|beat|kick|hard":   _make_bucket((0.0, 0.0)),
                "rock|beat|kick":        _make_bucket((0.0, 0.0)),
            },
            velocity_thresholds={"kick": (40.0, 80.0)},
        )
        bucket, level, _ = _lookup(profile, "rock", "beat", "kick", 100, grid_pos=3)
        assert level == 1
        assert bucket is not None

    def test_stratified_tier_grid_pos_miss_falls_to_unstratified_grid_pos(self):
        # tier+grid_pos miss → unstratified+grid_pos present → level 2.
        profile = LoadedProfile(
            buckets={
                "rock|beat|kick|5":  _make_bucket((1.0, 0.0), (2.0, 1.0), (-1.0, -1.0)),
                "rock|beat|kick|hard": _make_bucket((0.0, 0.0)),
            },
            velocity_thresholds={"kick": (40.0, 80.0)},
        )
        bucket, level, _ = _lookup(profile, "rock", "beat", "kick", 100, grid_pos=5)
        assert level == 2
        assert bucket is not None

    def test_stratified_both_grid_pos_miss_falls_to_tier(self):
        # Both grid_pos keys absent → tier-only key at level 3.
        profile = LoadedProfile(
            buckets={
                "rock|beat|kick|hard": _make_bucket((0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)),
            },
            velocity_thresholds={"kick": (40.0, 80.0)},
        )
        bucket, level, _ = _lookup(profile, "rock", "beat", "kick", 100, grid_pos=5)
        assert level == 3
        assert bucket is not None

    def test_unstratified_exact_with_grid_pos(self):
        # Hi-hat with grid_pos bucket present → level 1.
        profile = LoadedProfile(
            buckets={
                "rock|beat|hihat_closed|7": _make_bucket((1.0, 0.0), (2.0, 1.0), (-1.0, -1.0)),
                "rock|beat|hihat_closed":   _make_bucket((0.0, 0.0)),
            },
            velocity_thresholds={},
        )
        bucket, level, _ = _lookup(profile, "rock", "beat", "hihat_closed", 80, grid_pos=7)
        assert level == 1
        assert bucket is not None

    def test_unstratified_grid_pos_miss_falls_to_style(self):
        # grid_pos=2 absent → falls back to style bucket at level 2.
        profile = LoadedProfile(
            buckets={
                "rock|beat|hihat_closed|7": _make_bucket((1.0, 0.0), (2.0, 1.0), (-1.0, -1.0)),
                "rock|beat|hihat_closed":   _make_bucket((0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)),
            },
            velocity_thresholds={},
        )
        bucket, level, _ = _lookup(profile, "rock", "beat", "hihat_closed", 80, grid_pos=2)
        assert level == 2
        assert bucket is not None

    def test_grid_pos_none_preserves_existing_levels(self):
        # grid_pos=None: existing level numbering must be unchanged.
        # Stratified exact tier → level 1, tier-drop → level 2.
        profile = LoadedProfile(
            buckets={
                "rock|beat|kick|hard": _make_bucket((1.0, 0.0), (2.0, 1.0), (-1.0, -1.0)),
                "rock|beat|kick":      _make_bucket((0.0, 0.0)),
            },
            velocity_thresholds={"kick": (40.0, 80.0)},
        )
        _, level, _ = _lookup(profile, "rock", "beat", "kick", 100, grid_pos=None)
        assert level == 1  # rock|beat|kick|hard is level 1

    def test_grid_pos_none_tier_drop_is_level2(self):
        # Exact tier key absent, grid_pos=None → tier-drop at level 2.
        profile = LoadedProfile(
            buckets={
                "rock|beat|kick": _make_bucket((0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)),
            },
            velocity_thresholds={"kick": (40.0, 80.0)},
        )
        _, level, _ = _lookup(profile, "rock", "beat", "kick", 100, grid_pos=None)
        assert level == 2


# ---------------------------------------------------------------------------
# TestHumaniseMeterHandling
# ---------------------------------------------------------------------------

class TestHumaniseMeterHandling:
    def _midi_with_time_sig(self, numerator: int, denominator: int) -> mido.MidiFile:
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.MetaMessage(
            "time_signature",
            numerator=numerator,
            denominator=denominator,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0,
        ))
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=480))
        track.append(mido.MetaMessage("end_of_track", time=0))
        return mid

    def _grid_pos_profile(self) -> LoadedProfile:
        """Profile that contains a grid-position bucket — triggers the 4/4 check."""
        return LoadedProfile(
            buckets={"rock|beat|kick|hard|3": _make_bucket((0.0, 0.0), (1.0, 1.0), (-1.0, -1.0))},
            velocity_thresholds={},
        )

    def _plain_profile(self) -> LoadedProfile:
        """Profile with no grid-position buckets — 4/4 check must not fire."""
        return LoadedProfile(buckets={}, velocity_thresholds={})

    def test_three_four_accepted_with_grid_pos_profile(self, tmp_path):
        # Step 1 fix: a straight non-4/4 file must be ACCEPTED even when the profile
        # has grid-pos buckets — grid_pos=None routes to the non-positional fallback
        # (the 6/8 precedent). Constant +50ms fallback bucket → deterministic shift.
        mid = self._midi_with_time_sig(3, 4)
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        profile = LoadedProfile(
            buckets={
                "rock|beat|kick|hard|3": _make_bucket((0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)),
                "rock|beat|kick":        _make_bucket((50.0, 0.0), (50.0, 0.0), (50.0, 0.0)),
            },
            velocity_thresholds={},
        )
        humanise(inp, out, profile, seed=0)  # must not raise
        result = mido.MidiFile(str(out))
        for track in result.tracks:
            for msg in track:
                assert msg.time >= 0
        events = _collect_abs(result)
        note_on_abs = next(t for t, m in events if m.type == "note_on" and m.velocity > 0)
        # +50 ms at 120 BPM (500_000 µs/beat, PPQ 480) → round(50 / 1.0417) = 48 ticks
        expected = round(50.0 / (DEFAULT_TEMPO_US / DEFAULT_PPQ / 1000.0))
        assert note_on_abs == expected, (
            f"expected the non-positional fallback shift (+{expected} ticks), got {note_on_abs}"
        )

    def test_three_four_gridpos_only_profile_passes_through(self, tmp_path):
        # Grid-pos-ONLY profile + 3/4 file: grid_pos=None finds no bucket at any level
        # → note passes through untouched (no raise, no corruption).
        mid = self._midi_with_time_sig(3, 4)
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        humanise(inp, out, self._grid_pos_profile())  # must not raise
        result = mido.MidiFile(str(out))
        events = _collect_abs(result)
        note_on_abs, note_on_msg = next(
            (t, m) for t, m in events if m.type == "note_on" and m.velocity > 0
        )
        assert note_on_abs == 0
        assert note_on_msg.velocity == 80

    def test_mixed_non_six_eight_meters_accepted(self, tmp_path):
        # 4/4 followed by 3/4 (non-6/8 mix) → accepted with grid_pos=None; previously
        # raised when the profile had grid-pos buckets.
        mid = self._midi_with_time_sig(4, 4)
        mid.tracks[0].insert(2, mido.MetaMessage(
            "time_signature", numerator=3, denominator=4,
            clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0,
        ))
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        humanise(inp, out, self._grid_pos_profile())  # must not raise

    # --- helpers for 6/8 tests -------------------------------------------

    def _six_eight_midi_with_kick(self, note_delta_from_boundary: int = 10) -> mido.MidiFile:
        """6/8 file (time_sig at tick 0) with one kick note slightly off an 8th-note boundary."""
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.MetaMessage(
            "time_signature",
            numerator=6, denominator=8,
            clocks_per_click=24, notated_32nd_notes_per_beat=8,
            time=0,
        ))
        # ppq=480 → 8th note = 240 ticks; place note slightly off boundary 0
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80,
                                  time=note_delta_from_boundary))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=240))
        track.append(mido.MetaMessage("end_of_track", time=0))
        return mid

    def _profile_with_kick_fallback(self) -> LoadedProfile:
        """Grid-pos profile plus a non-positional 'rock|beat|kick' fallback with zero offset."""
        return LoadedProfile(
            buckets={
                "rock|beat|kick|hard|3": _make_bucket((0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)),
                "rock|beat|kick":        _make_bucket((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            },
            velocity_thresholds={},
        )

    def _profile_kick_zero_offset(self) -> LoadedProfile:
        """Plain profile with a zero-offset kick bucket — snapping to grid is the only movement."""
        return LoadedProfile(
            buckets={"rock|beat|kick": _make_bucket((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))},
            velocity_thresholds={},
        )

    # --- 6/8 tests -------------------------------------------------------

    def test_six_eight_allowed_with_gridpos_profile(self, tmp_path):
        # module 11: 6/8 (time_sig at tick 0) must no longer raise with grid-pos profile
        mid = self._six_eight_midi_with_kick()
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        humanise(inp, out, self._grid_pos_profile())   # must not raise

    def test_six_eight_snaps_to_eighth_grid(self, tmp_path):
        # Zero-offset profile — only movement is the grid snap; output tick must be
        # a multiple of ppq // 2 (= 240 ticks for ppq=480).
        mid = self._six_eight_midi_with_kick(note_delta_from_boundary=10)
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        humanise(inp, out, self._profile_kick_zero_offset(), seed=0)
        result = mido.MidiFile(str(out))
        eighth = DEFAULT_PPQ // 2
        abs_tick = 0
        note_ticks = []
        for msg in result.tracks[0]:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0 and msg.note == 36:
                note_ticks.append(abs_tick)
        assert note_ticks, "no kick note_on found in output"
        for t in note_ticks:
            assert t % eighth == 0, f"tick {t} is not on an 8th-note boundary (eighth={eighth})"

    def test_six_eight_uses_non_positional_fallback(self, tmp_path):
        # Grid-pos profile + 6/8 file → grid_pos=None → falls back to 'rock|beat|kick'
        mid = self._six_eight_midi_with_kick()
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        humanise(inp, out, self._profile_with_kick_fallback(), seed=0)  # must not raise

    def test_six_eight_mixed_raises(self, tmp_path):
        # 6/8 at tick 0 followed by a 4/4 event → mixed → ValueError
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.MetaMessage(
            "time_signature", numerator=6, denominator=8,
            clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0,
        ))
        track.append(mido.MetaMessage(
            "time_signature", numerator=4, denominator=4,
            clocks_per_click=24, notated_32nd_notes_per_beat=8, time=960,
        ))
        track.append(mido.MetaMessage("end_of_track", time=0))
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        with pytest.raises(ValueError, match="Mixed time signatures"):
            humanise(inp, out, self._plain_profile())

    def test_six_eight_late_start_raises(self, tmp_path):
        # Single 6/8 event not at tick 0 → implicit 4/4 prefix → mixed → ValueError
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO_US, time=0))
        track.append(mido.MetaMessage(
            "time_signature", numerator=6, denominator=8,
            clocks_per_click=24, notated_32nd_notes_per_beat=8, time=480,
        ))
        track.append(mido.MetaMessage("end_of_track", time=0))
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        with pytest.raises(ValueError, match="implicit 4/4"):
            humanise(inp, out, self._plain_profile())

    def test_three_four_accepted_with_plain_profile(self, tmp_path):
        # non-4/4 file + profile with NO grid-pos buckets → must not raise (regression guard)
        mid = self._midi_with_time_sig(3, 4)
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        humanise(inp, out, self._plain_profile())  # must not raise

    def test_explicit_four_four_accepted(self, tmp_path):
        mid = self._midi_with_time_sig(4, 4)
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        humanise(inp, out, self._grid_pos_profile())  # must not raise

    def test_no_time_sig_accepted(self, tmp_path):
        # No time_signature event → MIDI default 4/4 → accepted even with grid-pos profile.
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=80, time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
        ])
        inp = tmp_path / "in.mid"
        out = tmp_path / "out.mid"
        mid.save(str(inp))
        humanise(inp, out, self._grid_pos_profile())  # must not raise


# ---------------------------------------------------------------------------
# push vs no-push de-bias test
# ---------------------------------------------------------------------------

class TestPushFlag:
    """push=True preserves directional offset; push=False (default) removes it."""

    PPQ = 480
    TEMPO_US = 500_000  # 120 BPM; 1 tick = 500_000/480 µs ≈ 1.042 ms

    def _profile_with_known_mean(self) -> LoadedProfile:
        """Profile with a constant +50ms offset bucket and bucket_offset_means set."""
        # Three identical pairs → KDE is degenerate → falls back to uniform draw,
        # which always returns 50.0 ms. No randomness, no clamping surprise.
        offsets = np.array([50.0, 50.0, 50.0])
        vel_deltas = np.array([0.0, 0.0, 0.0])
        bucket = BucketProfile(offsets=offsets, vel_deltas=vel_deltas, kde=None)
        return LoadedProfile(
            buckets={"rock|beat|kick": bucket},
            velocity_thresholds={},
            bucket_offset_means={"rock|beat|kick": 50.0},
        )

    def _isolated_kick_midi(self, tmp_path: Path) -> Path:
        """Single kick at tick 4*PPQ (bar 2 beat 1) — no neighbours, wide legal window."""
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=self.TEMPO_US, time=0))
        # silence before the kick
        track.append(mido.Message("note_on",  channel=9, note=36, velocity=80, time=4 * self.PPQ))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0,  time=self.PPQ))
        track.append(mido.MetaMessage("end_of_track", time=0))
        p = tmp_path / "isolated_kick.mid"
        mid.save(str(p))
        return p

    def _kick_abs_tick(self, path: Path) -> int:
        mid = mido.MidiFile(str(path))
        abs_tick = 0
        for track in mid.tracks:
            for msg in track:
                abs_tick += msg.time
                if msg.type == "note_on" and msg.velocity > 0 and msg.note == 36:
                    return abs_tick
        raise AssertionError("no kick found in output")

    def test_push_true_applies_offset(self, tmp_path):
        inp = self._isolated_kick_midi(tmp_path)
        out = tmp_path / "out_push.mid"
        profile = self._profile_with_known_mean()

        humanise(inp, out, profile, seed=0, push=True)

        grid_tick = 4 * self.PPQ
        result_tick = self._kick_abs_tick(out)
        # +50ms at 120 BPM → +50 / (500_000/480/1000) ≈ +48 ticks
        expected_delta = round(50.0 / (self.TEMPO_US / self.PPQ / 1000.0))
        assert result_tick == grid_tick + expected_delta

    def test_push_false_removes_mean(self, tmp_path):
        inp = self._isolated_kick_midi(tmp_path)
        out = tmp_path / "out_nopush.mid"
        profile = self._profile_with_known_mean()

        humanise(inp, out, profile, seed=0, push=False)

        grid_tick = 4 * self.PPQ
        result_tick = self._kick_abs_tick(out)
        # offset_ms = 50.0 - mean(50.0) = 0.0 → no displacement
        assert result_tick == grid_tick

    def test_push_false_is_default(self, tmp_path):
        inp = self._isolated_kick_midi(tmp_path)
        out = tmp_path / "out_default.mid"
        profile = self._profile_with_known_mean()

        humanise(inp, out, profile, seed=0)  # no push kwarg

        grid_tick = 4 * self.PPQ
        assert self._kick_abs_tick(out) == grid_tick


# ---------------------------------------------------------------------------
# TestGrooveDrift — AR(1) clock + variance split (unit, no MIDI)
# ---------------------------------------------------------------------------

class TestGrooveDrift:
    def test_variance_preserved_across_phi(self):
        # For a stationary bucket, the sqrt(1-beta)/sqrt(beta) split keeps Var(output)==Var(c).
        rng = np.random.RandomState(0)
        c = rng.normal(0.0, 4.0, 40000)
        for phi in (0.3, 0.5, 0.8):
            np.random.seed(123)
            g = GrooveDrift(phi)
            out = np.array([g.step(x, 4.0) for x in c])
            assert out.var() == pytest.approx(c.var(), rel=0.06)

    def test_lag1_autocorr_is_one_minus_beta_times_phi(self):
        # The independent residual dilutes the drift's autocorrelation (phi) by (1-beta).
        rng = np.random.RandomState(1)
        c = rng.normal(0.0, 4.0, 60000)
        for phi in (0.3, 0.5, 0.8):
            np.random.seed(7)
            g = GrooveDrift(phi)
            out = np.array([g.step(x, 4.0) for x in c])
            ac = float(np.corrcoef(out[:-1], out[1:])[0, 1])
            assert ac == pytest.approx((1.0 - RESIDUAL_SHARE) * phi, abs=0.03)

    def test_ar_recursion_exact_with_zero_sigma(self):
        # sigma=0 removes residual randomness → deterministic; verify the AR recursion and
        # the sqrt(1-beta) drift weight exactly (fails for a wrong constant).
        phi = 0.5
        g = GrooveDrift(phi)
        innov = math.sqrt(1 - phi**2)
        wd = math.sqrt(1 - RESIDUAL_SHARE)
        d = 0.0
        for c in (10.0, -4.0, 7.0, 0.0):
            d = phi * d + innov * c
            assert g.step(c, 0.0) == pytest.approx(wd * d)

    def test_autocorr_zero_at_tiny_phi(self):
        rng = np.random.RandomState(2)
        c = rng.normal(0.0, 4.0, 40000)
        np.random.seed(3)
        g = GrooveDrift(1e-9)
        out = np.array([g.step(x, 4.0) for x in c])
        assert abs(float(np.corrcoef(out[:-1], out[1:])[0, 1])) < 0.03

    def test_drift_starts_at_zero(self):
        assert GrooveDrift(0.5).drift == 0.0


# ---------------------------------------------------------------------------
# TestCoupling — phi=0 independence vs phi>0 shared-nudge (incl. --push)
# ---------------------------------------------------------------------------

class TestCoupling:
    PPQ = 480
    TEMPO_US = 500_000  # 120 BPM → 1 tick ≈ 1.0417 ms

    def _chord_midi(self, tmp_path):
        # kick(36) then snare(38) on the SAME tick at bar 2 beat 1 (isolated, wide window).
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=self.TEMPO_US, time=0))
        G = 4 * self.PPQ
        tr.append(mido.Message("note_on",  channel=9, note=36, velocity=100, time=G))
        tr.append(mido.Message("note_on",  channel=9, note=38, velocity=100, time=0))
        # long notes (240 ticks) so the +10/+40 ms late shifts aren't clamped by note_off
        tr.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=240))
        tr.append(mido.Message("note_off", channel=9, note=38, velocity=0, time=0))
        tr.append(mido.MetaMessage("end_of_track", time=0))
        p = tmp_path / "chord.mid"; mid.save(str(p))
        return p

    def _note_ticks(self, path):
        mid = mido.MidiFile(str(path)); t = 0; out = {}
        for msg in mid.tracks[0]:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                out.setdefault(msg.note, []).append(t)
        return out

    def _gap_ms(self, path):
        mid = mido.MidiFile(str(path))
        tmap = build_tempo_map(mid)
        ticks = self._note_ticks(path)
        lo, hi = sorted((ticks[36][0], ticks[38][0]))
        return ticks_to_ms_with_map(lo, hi, tmap, self.PPQ)

    def _ms_per_tick(self):
        return self.TEMPO_US / self.PPQ / 1000.0

    # kick +10 ms, snare +40 ms (constant single-value buckets)
    def _profile_meanless(self):
        return LoadedProfile(
            buckets={
                "rock|beat|kick":  BucketProfile(np.array([10.0]), np.array([0.0]), None),
                "rock|beat|snare": BucketProfile(np.array([40.0]), np.array([0.0]), None),
            },
            velocity_thresholds={},
        )

    def _profile_with_means(self):
        p = self._profile_meanless()
        return LoadedProfile(
            buckets=p.buckets, velocity_thresholds={},
            bucket_offset_means={"rock|beat|kick": 10.0, "rock|beat|snare": 40.0},
        )

    def test_phi0_leaves_chord_members_independent(self, tmp_path):
        # Coupling OFF at phi=0: each lands at its OWN offset → clearly separated (~30 ms).
        inp = self._chord_midi(tmp_path); out = tmp_path / "o.mid"
        humanise(inp, out, self._profile_meanless(), seed=0, phi=0.0)
        assert self._gap_ms(out) == pytest.approx(30.0, abs=3.0)

    def test_coupled_hits_tight_phi_gt_0(self, tmp_path):
        inp = self._chord_midi(tmp_path); out = tmp_path / "o.mid"
        humanise(inp, out, self._profile_meanless(), seed=0, phi=0.5)
        assert self._gap_ms(out) <= 2 * COUPLED_RESIDUAL_MS + self._ms_per_tick()

    def test_coupled_hits_tight_under_push_with_differing_means(self, tmp_path):
        # Shared lean: kick mean=10, snare mean=40 differ, but under --push they still land
        # together (would separate by ~30 ms if each re-added its own lean).
        inp = self._chord_midi(tmp_path); out = tmp_path / "o.mid"
        humanise(inp, out, self._profile_with_means(), seed=0, phi=0.5, push=True)
        assert self._gap_ms(out) <= 2 * COUPLED_RESIDUAL_MS + self._ms_per_tick()

    def test_coupled_stays_tight_with_interspersed_fixed_note(self, tmp_path):
        # Regression: a non-shiftable note at the SAME tick BETWEEN chord members used to make
        # the coupled member reuse the anchor's pre-clamp desired offset and fly ~40 ms late
        # once the anchor was window-clamped by the fixed note. It must now track the anchor's
        # actual landing and stay tight (within a tick or two).
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=self.TEMPO_US, time=0))
        G = 4 * self.PPQ
        tr.append(mido.Message("note_on",  channel=9, note=36, velocity=100, time=G))  # kick (anchor)
        tr.append(mido.Message("note_on",  channel=0, note=60, velocity=100, time=0))  # unshiftable, same tick
        tr.append(mido.Message("note_on",  channel=9, note=38, velocity=100, time=0))  # snare (coupled)
        tr.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=240))
        tr.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=0))
        tr.append(mido.Message("note_off", channel=9, note=38, velocity=0, time=0))
        tr.append(mido.MetaMessage("end_of_track", time=0))
        inp = tmp_path / "chord_fixed.mid"; mid.save(str(inp))
        out = tmp_path / "o.mid"
        # kick & snare both want +40 ms late; the fixed note at G clamps the anchor to G-1.
        prof = LoadedProfile(
            buckets={
                "rock|beat|kick":  BucketProfile(np.array([40.0]), np.array([0.0]), None),
                "rock|beat|snare": BucketProfile(np.array([40.0]), np.array([0.0]), None),
            },
            velocity_thresholds={},
        )
        humanise(inp, out, prof, seed=0, phi=0.5, intensity=1.0)
        ticks = self._note_ticks(out)
        assert abs(ticks[36][0] - ticks[38][0]) <= 2   # tight (was ~38 ticks before the fix)


# ---------------------------------------------------------------------------
# TestLegacyProfileNoAmplification + velocity-only clock, phi validation
# ---------------------------------------------------------------------------

class TestGrooveDriftInteractions:
    PPQ = 480
    TEMPO_US = 500_000

    def _kick_line(self, tmp_path, n, step_ticks, dur=200):
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=self.TEMPO_US, time=0))
        prev = 0; origs = []
        for i in range(n):
            t = i * step_ticks
            tr.append(mido.Message("note_on",  channel=9, note=36, velocity=90, time=t - prev)); prev = t
            # note long enough that late timing shifts fit within the note (no note_off clamp)
            tr.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=dur)); prev = t + dur
            origs.append(t)
        tr.append(mido.MetaMessage("end_of_track", time=0))
        p = tmp_path / "kicks.mid"; mid.save(str(p))
        return p, origs

    def test_meanless_profile_lean_not_amplified(self, tmp_path):
        # Legacy/meanless profile: offsets mean +30 ms with spread; the systematic lean must
        # be preserved (~30 ms), NOT amplified to sqrt(1-phi^2)/(1-phi)*30 ≈ 52 ms.
        prof = LoadedProfile(
            buckets={"rock|beat|kick": BucketProfile(np.array([20.0, 40.0]),
                                                     np.array([0.0, 0.0]), None)},
            velocity_thresholds={},
        )
        inp, _ = self._kick_line(tmp_path, n=48, step_ticks=self.PPQ)
        out = tmp_path / "o.mid"
        humanise(inp, out, prof, seed=0, phi=0.5)
        mid = mido.MidiFile(str(out)); tmap = build_tempo_map(mid)
        offs = []; t = 0
        for msg in mid.tracks[0]:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                grid = (t // self.PPQ) * self.PPQ
                offs.append(ticks_to_ms_with_map(grid, t, tmap, self.PPQ))
        avg = float(np.mean(offs))
        assert avg == pytest.approx(30.0, abs=8.0)
        assert avg < 45.0   # firmly below the amplified ~52 ms

    def test_velocity_only_never_advances_clock(self, tmp_path):
        # velocity_only bypasses timing entirely — positions untouched even at high phi.
        prof = LoadedProfile(
            buckets={"rock|beat|kick": BucketProfile(np.array([20.0, -20.0, 5.0]),
                                                     np.array([10.0, -10.0, 0.0]), None)},
            velocity_thresholds={},
        )
        inp, origs = self._kick_line(tmp_path, n=8, step_ticks=240)
        out = tmp_path / "o.mid"
        humanise(inp, out, prof, seed=0, phi=0.9, velocity_only=True)
        res = mido.MidiFile(str(out)); t = 0; got = []
        for msg in res.tracks[0]:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                got.append(t)
        assert got == origs

    def test_sample_stream_independent_of_phi(self, tmp_path):
        # The residual RNG is separate from the global sample stream, so changing phi must
        # NOT change the drawn offset/velocity samples — only the timing *processing*. This
        # keeps a phi A/B a clean timing-only comparison.
        prof = LoadedProfile(
            buckets={"rock|beat|kick": BucketProfile(
                np.array([20.0, -15.0, 8.0, -3.0, 12.0]),
                np.array([10.0, -8.0, 4.0, -2.0, 6.0]), None)},
            velocity_thresholds={},
        )
        inp, _ = self._kick_line(tmp_path, n=16, step_ticks=self.PPQ)
        a, b = tmp_path / "a.mid", tmp_path / "b.mid"
        humanise(inp, a, prof, seed=5, phi=0.0)
        humanise(inp, b, prof, seed=5, phi=0.5)

        def vels(p):
            m = mido.MidiFile(str(p))
            return [msg.velocity for msg in m.tracks[0]
                    if msg.type == "note_on" and msg.velocity > 0]

        def ticks(p):
            m = mido.MidiFile(str(p)); t = 0; out = []
            for msg in m.tracks[0]:
                t += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    out.append(t)
            return out

        assert vels(a) == vels(b)      # velocities identical across phi
        assert ticks(a) != ticks(b)    # timing differs → drift is actually engaged

    @pytest.mark.parametrize("bad_phi", [1.0, 1.5, -0.1])
    def test_phi_out_of_range_raises(self, tmp_path, bad_phi):
        mid = _make_midi([
            mido.Message("note_on",  channel=9, note=36, velocity=80, time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
        ])
        inp = _save_load(mid, tmp_path); out = tmp_path / "o.mid"
        profs = load_profile(_write_profile(tmp_path, _profile_dict()))
        with pytest.raises(ValueError, match="phi"):
            humanise(inp, out, profs, phi=bad_phi)
