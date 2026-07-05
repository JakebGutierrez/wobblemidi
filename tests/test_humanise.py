"""Tests for pocketmidi/humanise.py"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mido
import numpy as np
import pytest

from pocketmidi.humanise import (
    COUPLED_RESIDUAL_MS,
    EPSILON_TICKS,
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
        out = _run_humanise(self._two_channel_midi(), self._PROFILE, tmp_path,
                            intensity=1.0, seed=0)
        ons = self._note_ons_by_channel(out)
        assert ons[0] == (0, 64), "channel-0 drum-range note must pass through untouched"
        assert ons[9] == (960 + self._SHIFT_TICKS, 74), "channel-9 kick must be humanised"

    def test_all_channels_reenables_other_channels(self, tmp_path):
        out = _run_humanise(self._two_channel_midi(), self._PROFILE, tmp_path,
                            intensity=1.0, seed=0, all_channels=True)
        ons = self._note_ons_by_channel(out)
        assert ons[0] == (0 + self._SHIFT_TICKS, 74), "channel-0 note must shift with all_channels"
        assert ons[9] == (960 + self._SHIFT_TICKS, 74)

    def test_defaults(self):
        import inspect
        params = inspect.signature(humanise).parameters
        assert params["all_channels"].default is False
        assert params["phi"].default == 0.4  # roadmap: phi default 0.5 → 0.4
        assert params["intensity"].default == 0.35  # ear-tested; 1.0 = raw GMD spread


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
        humanise(inp, out, profile, intensity=1.0, seed=0)  # must not raise
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

        humanise(inp, out, profile, intensity=1.0, seed=0, push=True)

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

    # NOTE (O5): the exact-constant locks previously here — output autocorr ==
    # (1-RESIDUAL_SHARE)*phi and the AR-recursion/sqrt(1-beta) arithmetic — were
    # retired with the module-13 rebuild and replaced by the invariants below
    # (monotonicity, variance budget, reproducibility, phi=0 bypass, velocity
    # de-bias). Exact recursion constants are implementation detail, not contract.

    def test_autocorr_monotone_in_phi(self):
        # More groove-tightness → more hit-to-hit timing memory. The exact value is
        # tuning; the ordering is the contract.
        rng = np.random.RandomState(1)
        c = rng.normal(0.0, 4.0, 60000)
        acs = []
        for phi in (0.1, 0.4, 0.8):
            np.random.seed(7)
            g = GrooveDrift(phi)
            out = np.array([g.step(x, 4.0) for x in c])
            acs.append(float(np.corrcoef(out[:-1], out[1:])[0, 1]))
        assert acs[0] < acs[1] < acs[2]
        assert acs[0] > 0.02          # even small phi leaves some memory
        assert acs[2] < 0.85          # residual keeps it strictly below 1

    def test_deterministic_and_seed_dependent(self):
        # Same rng seed → identical step sequence; different seed → different residuals.
        c = [10.0, -4.0, 7.0, 0.0, 3.0]
        g1 = GrooveDrift(0.5, np.random.RandomState(9))
        g2 = GrooveDrift(0.5, np.random.RandomState(9))
        g3 = GrooveDrift(0.5, np.random.RandomState(10))
        s1 = [g1.step(x, 2.0) for x in c]
        s2 = [g2.step(x, 2.0) for x in c]
        s3 = [g3.step(x, 2.0) for x in c]
        assert s1 == s2
        assert s1 != s3

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
        humanise(inp, out, self._profile_meanless(), intensity=1.0, seed=0, phi=0.0)
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
# TestKitWideClock — groove clock + coupling shared across tracks
# ---------------------------------------------------------------------------

class TestKitWideClock:
    """The groove clock and chord coupling are kit-wide, not per-track.

    Multi-track drum MIDI (kick/snare on separate tracks — the normal DAW
    export) used to give each track its own drift clock and never coupled
    cross-track same-tick hits: measured flams up to ~73 ms at phi=0.5.
    """

    PPQ = 480
    TEMPO_US = 500_000  # 120 BPM → 1 tick ≈ 1.0417 ms
    N_HITS = 16

    def _two_track_midi(self, tmp_path) -> Path:
        """kick(36) on track 1, snare(38) on track 2, notated on the SAME ticks."""
        mid = mido.MidiFile(type=1, ticks_per_beat=self.PPQ)
        conductor = mido.MidiTrack(); mid.tracks.append(conductor)
        conductor.append(mido.MetaMessage("set_tempo", tempo=self.TEMPO_US, time=0))
        conductor.append(mido.MetaMessage("end_of_track", time=0))
        for note in (36, 38):
            tr = mido.MidiTrack(); mid.tracks.append(tr)
            prev = 0
            for i in range(self.N_HITS):
                t = i * self.PPQ
                tr.append(mido.Message("note_on", channel=9, note=note,
                                       velocity=100, time=t - prev))
                tr.append(mido.Message("note_off", channel=9, note=note,
                                       velocity=0, time=200))
                prev = t + 200
            tr.append(mido.MetaMessage("end_of_track", time=0))
        p = tmp_path / "two_track.mid"
        mid.save(str(p))
        return p

    def _profile(self) -> LoadedProfile:
        # Varied, meanless buckets so the drift clock is actually exercised.
        return LoadedProfile(
            buckets={
                "rock|beat|kick":  BucketProfile(
                    np.array([-20.0, -10.0, 0.0, 10.0, 20.0]), np.zeros(5), None),
                "rock|beat|snare": BucketProfile(
                    np.array([-30.0, -15.0, 0.0, 15.0, 30.0]), np.zeros(5), None),
            },
            velocity_thresholds={},
        )

    def _note_ticks(self, path, note) -> list[int]:
        mid = mido.MidiFile(str(path))
        out = []
        for tr in mid.tracks:
            t = 0
            for msg in tr:
                t += msg.time
                if msg.type == "note_on" and msg.velocity > 0 and msg.note == note:
                    out.append(t)
        return sorted(out)

    def test_cross_track_same_tick_hits_land_together(self, tmp_path):
        # The 73 ms case: at phi=0.5 every kick/snare pair notated on the same tick
        # must land within 2 ticks (~2 ms), coupled across tracks via the shared
        # anchor. With per-track clocks this input flammed up to ~42 ticks.
        inp = self._two_track_midi(tmp_path)
        out = tmp_path / "out.mid"
        humanise(inp, out, self._profile(), seed=3, phi=0.5)
        kicks = self._note_ticks(out, 36)
        snares = self._note_ticks(out, 38)
        assert len(kicks) == len(snares) == self.N_HITS
        # Windowing preserves per-track hit order, so pair i-th kick with i-th snare.
        gaps = [abs(k - s) for k, s in zip(kicks, snares)]
        assert max(gaps) <= 2, f"cross-track flam: gaps {gaps}"
        # Guard: the clock must actually be moving hits, not holding everything on grid.
        assert any(t != i * self.PPQ for i, t in enumerate(kicks))


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
        humanise(inp, out, prof, intensity=1.0, seed=0, phi=0.5)
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


# ---------------------------------------------------------------------------
# TestEngineTimingInvariants — O5 replacements measured through humanise()
# ---------------------------------------------------------------------------

class TestEngineTimingInvariants:
    """phi=0 bypass, autocorr monotone in phi, and the total-variance budget,
    asserted on actual engine output rather than on recursion constants."""

    PPQ = 480
    TEMPO_US = 500_000  # 120 BPM → 1 tick ≈ 1.0417 ms

    def _kick_line(self, tmp_path, n):
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=self.TEMPO_US, time=0))
        prev = 0
        for i in range(n):
            t = i * self.PPQ
            tr.append(mido.Message("note_on",  channel=9, note=36, velocity=90, time=t - prev))
            tr.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=200))
            prev = t + 200
        tr.append(mido.MetaMessage("end_of_track", time=0))
        p = tmp_path / "kicks.mid"; mid.save(str(p))
        return p

    def _profile(self):
        # meanless, spread-y offsets (±20 ms ≈ ±19 ticks: far inside every window)
        return LoadedProfile(
            buckets={"rock|beat|kick": BucketProfile(
                np.array([-20.0, -10.0, 0.0, 10.0, 20.0]), np.zeros(5), None)},
            velocity_thresholds={},
        )

    def _offsets_ticks(self, path):
        mid = mido.MidiFile(str(path))
        t = 0; offs = []
        for msg in mid.tracks[0]:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                grid = round(t / self.PPQ) * self.PPQ
                offs.append(t - grid)
        return np.array(offs, dtype=float)

    def _run(self, tmp_path, phi, n=1200, seed=11):
        inp = self._kick_line(tmp_path, n)
        out = tmp_path / f"out_phi{phi}.mid"
        humanise(inp, out, self._profile(), seed=seed, phi=phi, timing_only=True)
        return self._offsets_ticks(out)

    def test_phi0_bypass_no_timing_memory(self, tmp_path):
        # phi=0: drift inert — successive offsets are independent draws.
        offs = self._run(tmp_path, phi=0.0)
        r = float(np.corrcoef(offs[:-1], offs[1:])[0, 1])
        assert abs(r) < 0.06

    def test_autocorr_monotone_in_phi_through_engine(self, tmp_path):
        rs = []
        for phi in (0.0, 0.4, 0.8):
            offs = self._run(tmp_path, phi=phi)
            rs.append(float(np.corrcoef(offs[:-1], offs[1:])[0, 1]))
        assert rs[0] < rs[1] < rs[2]
        assert rs[1] > 0.15   # default phi leaves visible pocket memory

    def test_total_variance_budget_across_phi(self, tmp_path):
        # Drift redistributes timing variance over time; it must not add or remove it.
        # Same seed → identical sample draws across phi, so this comparison is clean.
        sd0 = self._run(tmp_path, phi=0.0).std()
        sd5 = self._run(tmp_path, phi=0.5).std()
        assert sd5 == pytest.approx(sd0, rel=0.12)


# ---------------------------------------------------------------------------
# Module 13 (spec B2) — kick-only velocity drift
# ---------------------------------------------------------------------------

class TestVelDrift:
    PPQ = 480
    TEMPO_US = 500_000

    def _line(self, tmp_path, note, n, vel=90):
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=self.TEMPO_US, time=0))
        prev = 0
        for i in range(n):
            t = i * self.PPQ
            tr.append(mido.Message("note_on",  channel=9, note=note, velocity=vel, time=t - prev))
            tr.append(mido.Message("note_off", channel=9, note=note, velocity=0, time=200))
            prev = t + 200
        tr.append(mido.MetaMessage("end_of_track", time=0))
        p = tmp_path / f"line{note}.mid"; mid.save(str(p))
        return p

    def _out_vels(self, path, note):
        mid = mido.MidiFile(str(path))
        return np.array([m.velocity for m in mid.tracks[0]
                         if m.type == "note_on" and m.velocity > 0 and m.note == note],
                        dtype=float)

    def _spready_profile(self, key):
        # zero-mean spread-y vel_deltas, zero offsets (timing never interferes)
        return LoadedProfile(
            buckets={key: BucketProfile(np.zeros(5),
                                        np.array([-12.0, -6.0, 0.0, 6.0, 12.0]), None)},
            velocity_thresholds={},
        )

    def test_kick_velocities_drift_snare_stays_white(self, tmp_path):
        # B2: kick gets the AR(1) velocity clock; snare (measured ~white) must not.
        n = 2000
        k_out, s_out = tmp_path / "k.mid", tmp_path / "s.mid"
        humanise(self._line(tmp_path, 36, n), k_out,
                 self._spready_profile("rock|beat|kick"), seed=0, velocity_only=True)
        humanise(self._line(tmp_path, 38, n), s_out,
                 self._spready_profile("rock|beat|snare"), seed=0, velocity_only=True)
        kv = self._out_vels(k_out, 36)
        sv = self._out_vels(s_out, 38)
        r_kick = float(np.corrcoef(kv[:-1], kv[1:])[0, 1])
        r_snare = float(np.corrcoef(sv[:-1], sv[1:])[0, 1])
        assert r_kick > 0.15, f"kick velocity shows no drift (r={r_kick:.3f})"
        assert abs(r_snare) < 0.08, f"snare velocity is not i.i.d. (r={r_snare:.3f})"

    def test_kick_velocity_variance_preserved(self, tmp_path):
        # The drift split redistributes velocity variance; it must not change the spread.
        out = tmp_path / "o.mid"
        humanise(self._line(tmp_path, 36, 4000), out,
                 self._spready_profile("rock|beat|kick"),
                 intensity=1.0, seed=1, velocity_only=True)
        kv = self._out_vels(out, 36) - 90.0
        raw_sd = float(np.array([-12.0, -6.0, 0.0, 6.0, 12.0]).std())
        assert float(kv.std()) == pytest.approx(raw_sd, rel=0.15)

    def test_legacy_biased_bucket_applied_statically(self, tmp_path):
        # Old-schema bucket with vel_delta mean +10 (median-based deltas): the bias must
        # come through as a static +10 level shift, never amplified by the AR recursion —
        # same backward-compat guard as the timing clock's mean-centring.
        prof = LoadedProfile(
            buckets={"rock|beat|kick": BucketProfile(
                np.zeros(5), np.array([6.0, 8.0, 10.0, 12.0, 14.0]), None)},
            velocity_thresholds={},
        )
        out = tmp_path / "o.mid"
        humanise(self._line(tmp_path, 36, 2000), out, prof,
                 intensity=1.0, seed=2, velocity_only=True)
        assert float(self._out_vels(out, 36).mean()) == pytest.approx(100.0, abs=1.0)

    def test_timing_identical_across_velocity_processing(self, tmp_path):
        # The velocity clock draws from its own RNG stream, so toggling velocity
        # processing must not change timing for the same seed (module-7 contract).
        prof = LoadedProfile(
            buckets={"rock|beat|kick": BucketProfile(
                np.array([-15.0, -5.0, 5.0, 15.0, 0.0]),
                np.array([-12.0, -6.0, 0.0, 6.0, 12.0]), None)},
            velocity_thresholds={},
        )
        inp = self._line(tmp_path, 36, 64)
        a, b = tmp_path / "a.mid", tmp_path / "b.mid"
        humanise(inp, a, prof, seed=5, timing_only=True)
        humanise(inp, b, prof, seed=5)

        def ticks(p):
            mid = mido.MidiFile(str(p)); t = 0; out = []
            for msg in mid.tracks[0]:
                t += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    out.append(t)
            return out

        assert ticks(a) == ticks(b)


# ---------------------------------------------------------------------------
# Module 13 (spec B4) — relative velocity tiering
# ---------------------------------------------------------------------------

from pocketmidi.humanise import (  # noqa: E402  (grouped with the tests that use them)
    RELATIVE_TIER_MIN_HITS,
    RELATIVE_TIER_MIN_SPREAD,
    _file_tier_thresholds,
)


class TestRelativeTierThresholds:
    ABSOLUTE = (57.0, 80.0)   # GMD-ish kick tertiles

    def test_too_few_hits_falls_back_to_absolute(self):
        v = [40, 100] * ((RELATIVE_TIER_MIN_HITS - 1) // 2)
        assert _file_tier_thresholds(v, self.ABSOLUTE) == self.ABSOLUTE

    def test_narrow_spread_falls_back_to_absolute(self):
        v = [90, 92, 94, 95, 96, 97, 99, 100] * 4   # p90-p10 < 12
        assert _file_tier_thresholds(v, self.ABSOLUTE) == self.ABSOLUTE

    def test_all_one_velocity_falls_back_to_absolute(self):
        assert _file_tier_thresholds([88] * 32, self.ABSOLUTE) == self.ABSOLUTE

    def test_continuous_spread_gives_relative_tertiles(self):
        v = list(range(60, 108, 3)) * 3   # even spread, no dominant gap
        low, high = _file_tier_thresholds(v, self.ABSOLUTE)
        assert (low, high) != self.ABSOLUTE
        assert 60 < low < high < 108
        np.testing.assert_allclose([low, high], np.percentile(v, [33, 66]))

    def test_two_cluster_maps_soft_hard_no_medium(self):
        # ghost/accent part: 30% ghosts at 25-30, 70% accents at 95-105
        v = [25, 30] * 6 + [95, 100, 105] * 10
        low, high = _file_tier_thresholds(v, self.ABSOLUTE)
        assert low == high   # collapsed boundary → medium is unreachable
        assert 30 < low < 95
        assert _velocity_tier(30, (low, high)) == "soft"
        assert _velocity_tier(95, (low, high)) == "hard"

    def test_two_level_palette_is_two_cluster(self):
        # exactly two programmed values (2-level palette) → soft/hard split
        v = [60] * 10 + [100] * 10
        low, high = _file_tier_thresholds(v, self.ABSOLUTE)
        assert low == high == pytest.approx(80.0)

    def test_imbalanced_two_value_ghosts_go_soft(self):
        # Codex review fix: [35]*14 + [110]*2 failed the 15% fraction gate, fell
        # through to tertiles that collapsed to (35, 35), and routed the GHOSTS to
        # hard. Two distinct values with a real spread must split soft/hard.
        v = [35] * 14 + [110] * 2
        low, high = _file_tier_thresholds(v, self.ABSOLUTE)
        assert _velocity_tier(35, (low, high)) == "soft"
        assert _velocity_tier(110, (low, high)) == "hard"

    def test_imbalanced_two_value_accent_majority(self):
        # mirror image: 2 ghosts against 14 accents
        v = [35] * 2 + [110] * 14
        low, high = _file_tier_thresholds(v, self.ABSOLUTE)
        assert _velocity_tier(35, (low, high)) == "soft"
        assert _velocity_tier(110, (low, high)) == "hard"

    def test_collapsed_tertiles_dominant_middle_stays_medium(self):
        # three levels with a dominant middle: both tertiles collapse onto 75;
        # tie-aware re-anchoring must keep soft/medium/hard intact
        v = [30] * 2 + [75] * 12 + [110] * 2
        low, high = _file_tier_thresholds(v, self.ABSOLUTE)
        assert _velocity_tier(30, (low, high)) == "soft"
        assert _velocity_tier(75, (low, high)) == "medium"
        assert _velocity_tier(110, (low, high)) == "hard"

    def test_collapsed_tertiles_dominant_bottom_goes_soft(self):
        # dominant duplicated value at the BOTTOM of a 3-level part, minority too
        # small for the two-cluster gap path (0.111 < 0.15) AND close neighbour so
        # the gap check fails: tertiles collapse onto 35 → it must stay soft
        v = [35] * 14 + [50] * 2 + [110] * 2
        low, high = _file_tier_thresholds(v, self.ABSOLUTE)
        assert _velocity_tier(35, (low, high)) == "soft"
        assert _velocity_tier(50, (low, high)) == "hard"
        assert _velocity_tier(110, (low, high)) == "hard"

    def test_tiny_minority_cluster_not_two_cluster(self):
        # one stray accent in 40 hits (<15%) must not force a two-cluster split
        v = [70 + (i % 5) * 4 for i in range(39)] + [120]
        low, high = _file_tier_thresholds(v, self.ABSOLUTE)
        assert low != high   # tertile path

    def test_ties_share_a_tier(self):
        v = [40] * 10 + [77] * 10 + [110] * 10
        low, high = _file_tier_thresholds(v, self.ABSOLUTE)
        tiers = {x: _velocity_tier(x, (low, high)) for x in (40, 77, 110)}
        assert len(set(tiers.values())) == len(tiers)   # distinct values, distinct tiers

    def test_spread_gate_boundary(self):
        # exactly at the p90-p10 threshold → relative tiering allowed
        v = ([80] * 10 + [80 + RELATIVE_TIER_MIN_SPREAD] * 10)
        low, high = _file_tier_thresholds(v, self.ABSOLUTE)
        assert low == high   # two-cluster relative split, not absolute


class TestRelativeTieringIntegration:
    PPQ = 480

    def test_relative_tiers_route_to_different_buckets(self, tmp_path):
        """A part whose velocities all sit inside ONE absolute tier must still
        route its low cluster to soft and high cluster to hard (tier-collapse fix).
        Bucket vel_deltas differ per tier, so output velocities reveal the routing."""
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
        prev = 0
        vels = [60, 100] * 8   # two-cluster, 16 hits, spread 40
        for i, v in enumerate(vels):
            t = i * self.PPQ
            tr.append(mido.Message("note_on",  channel=9, note=38, velocity=v, time=t - prev))
            tr.append(mido.Message("note_off", channel=9, note=38, velocity=0, time=200))
            prev = t + 200
        tr.append(mido.MetaMessage("end_of_track", time=0))
        inp = tmp_path / "in.mid"; mid.save(str(inp))

        # ABSOLUTE thresholds put EVERY hit (60 and 100) in "soft" (both < 110).
        # Relative tiering must split them 60→soft, 100→hard.
        prof = LoadedProfile(
            buckets={
                "rock|beat|snare|soft": BucketProfile(np.zeros(1), np.array([-20.0]), None),
                "rock|beat|snare|hard": BucketProfile(np.zeros(1), np.array([+20.0]), None),
            },
            velocity_thresholds={"snare": (110.0, 120.0)},
        )
        out = tmp_path / "o.mid"
        humanise(inp, out, prof, intensity=1.0, seed=0, velocity_only=True)

        got = [m.velocity for m in mido.MidiFile(str(out)).tracks[0]
               if m.type == "note_on" and m.velocity > 0]
        assert got == [40, 120] * 8   # 60-20 (soft bucket) / 100+20 (hard bucket)

    def test_absolute_fallback_when_no_evidence(self, tmp_path):
        """Same setup but all hits at one velocity → absolute thresholds apply
        (everything routes to the absolute tier for velocity 100: 'soft')."""
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
        prev = 0
        for i in range(16):
            t = i * self.PPQ
            tr.append(mido.Message("note_on",  channel=9, note=38, velocity=100, time=t - prev))
            tr.append(mido.Message("note_off", channel=9, note=38, velocity=0, time=200))
            prev = t + 200
        tr.append(mido.MetaMessage("end_of_track", time=0))
        inp = tmp_path / "in.mid"; mid.save(str(inp))

        prof = LoadedProfile(
            buckets={
                "rock|beat|snare|soft": BucketProfile(np.zeros(1), np.array([-20.0]), None),
                "rock|beat|snare|hard": BucketProfile(np.zeros(1), np.array([+20.0]), None),
            },
            velocity_thresholds={"snare": (110.0, 120.0)},
        )
        out = tmp_path / "o.mid"
        humanise(inp, out, prof, intensity=1.0, seed=0, velocity_only=True)
        got = [m.velocity for m in mido.MidiFile(str(out)).tracks[0]
               if m.type == "note_on" and m.velocity > 0]
        assert got == [80] * 16   # all soft: 100 - 20


# ---------------------------------------------------------------------------
# TestLeanAndPerGroupIntensity — GUI round 3 engine params
# push_amount (continuous lean, mirror-not-drag) and intensity_by_group
# (per-lane output gain on ONE shared clock, min-eff chord governance).
# Invariants and equivalences, not constant locks. Profiles carry NONZERO
# stored means — a zero-mean profile passes the lean matrix vacuously.
# ---------------------------------------------------------------------------

class TestLeanAndPerGroupIntensity:
    PPQ = 480
    TEMPO_US = 500_000               # 120 BPM → 1 tick ≈ 1.0417 ms
    MS_PER_TICK = 500_000 / 480 / 1000.0

    # -- builders -----------------------------------------------------------

    def _prof(self, kick_m=50.0, hat_m=-30.0, spread=True, means=True,
              snare_m=None) -> LoadedProfile:
        def bucket(m):
            if spread:
                offs = np.array([-20.0, -10.0, 0.0, 10.0, 20.0]) + m
                vds = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
            else:
                offs = np.array([m, m, m])
                vds = np.zeros(3)
            return BucketProfile(offsets=offs, vel_deltas=vds, kde=None)
        buckets = {
            "rock|beat|kick": bucket(kick_m),
            "rock|beat|hihat_closed": bucket(hat_m),
        }
        if snare_m is not None:
            buckets["rock|beat|snare"] = bucket(snare_m)
        means_d = ({k: float(np.mean(b.offsets)) for k, b in buckets.items()}
                   if means else {})
        return LoadedProfile(buckets=buckets, velocity_thresholds={},
                             bucket_offset_means=means_d)

    def _two_lane_midi(self, tmp_path, same_tick=True, n=12, hat_first=False,
                       name="two_lane.mid") -> Path:
        """Type 1: kick track + hat track. same_tick=True notates chords."""
        mid = mido.MidiFile(type=1, ticks_per_beat=self.PPQ)
        cond = mido.MidiTrack(); mid.tracks.append(cond)
        cond.append(mido.MetaMessage("set_tempo", tempo=self.TEMPO_US, time=0))
        cond.append(mido.MetaMessage("end_of_track", time=0))
        hat_shift = 0 if same_tick else self.PPQ // 2
        specs = [(42, 60, hat_shift), (36, 100, 0)] if hat_first else \
                [(36, 100, 0), (42, 60, hat_shift)]
        for note, vel, shift in specs:
            tr = mido.MidiTrack(); mid.tracks.append(tr)
            prev = 0
            for i in range(n):
                t = 4 * self.PPQ + i * self.PPQ + shift
                tr.append(mido.Message("note_on", channel=9, note=note,
                                       velocity=vel, time=t - prev))
                tr.append(mido.Message("note_off", channel=9, note=note,
                                       velocity=0, time=200))
                prev = t + 200
            tr.append(mido.MetaMessage("end_of_track", time=0))
        p = tmp_path / name
        mid.save(str(p))
        return p

    def _kick_line(self, tmp_path, n=8) -> Path:
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=self.TEMPO_US, time=0))
        prev = 0
        for i in range(n):
            t = 4 * self.PPQ * (i + 1)
            tr.append(mido.Message("note_on", channel=9, note=36, velocity=100,
                                   time=t - prev))
            tr.append(mido.Message("note_off", channel=9, note=36, velocity=0,
                                   time=200))
            prev = t + 200
        tr.append(mido.MetaMessage("end_of_track", time=0))
        p = tmp_path / "kicks.mid"
        mid.save(str(p))
        return p

    def _note_events(self, path, note):
        """Sorted [(abs_tick, velocity), ...] of note_ons for `note`."""
        out = []
        for tr in mido.MidiFile(str(path)).tracks:
            t = 0
            for msg in tr:
                t += msg.time
                if msg.type == "note_on" and msg.velocity > 0 and msg.note == note:
                    out.append((t, msg.velocity))
        return sorted(out)

    # -- regression vs the PRE-round-3 engine (real historical bytes) ------------

    def test_pre_round3_fixture_regression(self, tmp_path):
        """Current engine reproduces the pre-round-3 engine byte-for-byte at the
        push endpoints, via BOTH spellings (push bool and push_amount).

        Fixtures in tests/fixtures/pre_r3_push{0,1}_phi{00,05}.mid were generated
        by running humanise() from commit 63862a1^ (e9401a4 — before push_amount,
        intensity_by_group, and windowed coupling) on exactly the builders below
        with seed 42, intensity 1.0. A self-comparison (push vs push_amount, both
        through the current code) cannot catch semantic drift — this can.

        The fixture input is multi-track with SAME-TICK chords every beat, so
        this also locks that time-windowed coupling leaves zero-gap (same-tick)
        clusters and solo hits byte-identical, at phi=0 AND phi>0.
        """
        prof = self._prof()
        inp = self._two_lane_midi(tmp_path, same_tick=True)
        fixdir = Path(__file__).parent / "fixtures"
        for push in (False, True):
            for phi in (0.0, 0.5):
                fixture = (fixdir /
                           f"pre_r3_push{int(push)}_phi{str(phi).replace('.', '')}.mid")
                expected = fixture.read_bytes()
                a = tmp_path / "a.mid"
                humanise(inp, a, prof, intensity=1.0, seed=42, phi=phi, push=push)
                assert a.read_bytes() == expected, ("push bool drifted", push, phi)
                b = tmp_path / "b.mid"
                humanise(inp, b, prof, intensity=1.0, seed=42, phi=phi,
                         push_amount=1.0 if push else 0.0)
                assert b.read_bytes() == expected, ("push_amount drifted", push, phi)

    def test_intensity_by_group_none_empty_identical(self, tmp_path):
        prof = self._prof()
        inp = self._two_lane_midi(tmp_path, same_tick=True)
        for phi in (0.0, 0.5):
            outs = []
            for i, kw in enumerate([dict(), dict(intensity_by_group=None),
                                    dict(intensity_by_group={})]):
                o = tmp_path / f"o{i}.mid"
                humanise(inp, o, prof, intensity=1.0, seed=7, phi=phi, **kw)
                outs.append(o.read_bytes())
            assert outs[0] == outs[1] == outs[2]

    # -- lean semantics --------------------------------------------------------

    def test_lean_mirror_and_linearity(self, tmp_path):
        """Mean landing at a=-1 reflects a=+1 about the a=0 mean; a=0.5 is linear.

        ms-level with tolerance (tick rounding), not per-hit byte equality.
        Constant-offset bucket (m=+50, stored mean 50) → deterministic even at
        phi>0 (centred sample and residual sigma are both zero). Runs on BOTH
        the phi==0 exact-bypass path and the phi>0 drift path — the two de-bias
        sites are separate code.
        """
        prof = self._prof(kick_m=50.0, spread=False)
        inp = self._kick_line(tmp_path)
        grid = [4 * self.PPQ * (i + 1) for i in range(8)]

        def mean_delta_ms(amount, phi):
            out = tmp_path / f"lean_{amount}_{phi}.mid"
            humanise(inp, out, prof, intensity=1.0, seed=5, phi=phi,
                     push_amount=amount)
            ticks = [t for t, _v in self._note_events(out, 36)]
            deltas = [(t - g) * self.MS_PER_TICK for t, g in zip(ticks, grid)]
            return float(np.mean(deltas))

        for phi in (0.0, 0.4):
            d_pos = mean_delta_ms(1.0, phi)
            d_neg = mean_delta_ms(-1.0, phi)
            d_zero = mean_delta_ms(0.0, phi)
            d_half = mean_delta_ms(0.5, phi)
            tol = 2.0 * self.MS_PER_TICK
            assert d_pos == pytest.approx(50.0, abs=tol), phi
            assert d_zero == pytest.approx(0.0, abs=tol), phi
            # reflection about the a=0 mean
            assert d_neg - d_zero == pytest.approx(-(d_pos - d_zero), abs=tol), phi
            # linearity spot check
            assert d_half == pytest.approx(d_zero + 0.5 * (d_pos - d_zero), abs=tol), phi

    def test_legacy_profile_lean_noop(self, tmp_path, capsys):
        """No stored means → any lean is a byte-identical no-op, with a stderr note."""
        prof = self._prof(means=False)
        inp = self._two_lane_midi(tmp_path, same_tick=False)
        outs = {}
        for amount in (None, 1.0, 0.5, -1.0):
            o = tmp_path / f"legacy_{amount}.mid"
            kw = {} if amount is None else {"push_amount": amount}
            capsys.readouterr()
            humanise(inp, o, prof, intensity=1.0, seed=9, phi=0.5, **kw)
            err = capsys.readouterr().err
            if amount in (0.5, -1.0):
                assert "no per-bucket lean means" in err
            else:
                assert "no per-bucket lean means" not in err
            outs[amount] = o.read_bytes()
        assert len(set(outs.values())) == 1

    # -- per-group intensity semantics ---------------------------------------------

    def test_chord_min_eff_governs_both_orders(self, tmp_path):
        """Same-tick kick+hat with {kick: 0.15, hats: 0.8}: the pair lands together
        AND at the min-eff (0.15) displacement — the tightest limb is the
        timekeeper. Locked in BOTH track/stream orders (constant-offset profile
        makes landings order-independent)."""
        prof = self._prof(kick_m=50.0, hat_m=50.0, spread=False)
        scales = {"kick": 0.15, "hihat_closed": 0.8}
        expected_ticks = round((50.0 * 0.15) / self.MS_PER_TICK)   # ≈ 7
        wrong_ticks = round((50.0 * 0.8) / self.MS_PER_TICK)       # ≈ 38
        for hat_first in (False, True):
            inp = self._two_lane_midi(tmp_path, same_tick=True, hat_first=hat_first,
                                      name=f"chord_{hat_first}.mid")
            out = tmp_path / f"chord_out_{hat_first}.mid"
            humanise(inp, out, prof, intensity=1.0, seed=3, phi=0.5,
                     push_amount=1.0, intensity_by_group=scales)
            kicks = [t for t, _ in self._note_events(out, 36)]
            hats = [t for t, _ in self._note_events(out, 42)]
            grid = [4 * self.PPQ + i * self.PPQ for i in range(12)]
            gaps = [abs(k - h) for k, h in zip(kicks, hats)]
            assert max(gaps) <= 2, f"chord flam (hat_first={hat_first}): {gaps}"
            disp = [k - g for k, g in zip(kicks, grid)]
            for d in disp:
                assert abs(d - expected_ticks) <= 2, (
                    f"anchor not governed by min eff: displacement {d} ticks, "
                    f"expected ~{expected_ticks} (0.8-eff would be ~{wrong_ticks})")

    def test_chord_min_eff_three_lanes(self, tmp_path):
        """MIN over ALL members of a 3-lane chord — the tightest lane governs.

        The two-lane version would pass a broken pairwise/adjacent-min
        implementation; with three lanes the minimum (snare, 0.2) is neither the
        loudest, the first, nor the last member.
        """
        prof = self._prof(kick_m=50.0, hat_m=50.0, snare_m=50.0, spread=False)
        scales = {"kick": 0.5, "snare": 0.2, "hihat_closed": 0.8}
        mid = mido.MidiFile(type=1, ticks_per_beat=self.PPQ)
        cond = mido.MidiTrack(); mid.tracks.append(cond)
        cond.append(mido.MetaMessage("set_tempo", tempo=self.TEMPO_US, time=0))
        cond.append(mido.MetaMessage("end_of_track", time=0))
        for note, vel in ((36, 110), (38, 90), (42, 60)):
            tr = mido.MidiTrack(); mid.tracks.append(tr)
            prev = 0
            for i in range(8):
                t = 4 * self.PPQ + i * self.PPQ
                tr.append(mido.Message("note_on", channel=9, note=note,
                                       velocity=vel, time=t - prev))
                tr.append(mido.Message("note_off", channel=9, note=note,
                                       velocity=0, time=200))
                prev = t + 200
            tr.append(mido.MetaMessage("end_of_track", time=0))
        inp = tmp_path / "three_lane.mid"
        mid.save(str(inp))

        out = tmp_path / "out.mid"
        humanise(inp, out, prof, intensity=1.0, seed=3, phi=0.5,
                 push_amount=1.0, intensity_by_group=scales)
        expected = round((50.0 * 0.2) / self.MS_PER_TICK)   # min eff governs: ~10
        grid = [4 * self.PPQ + i * self.PPQ for i in range(8)]
        for note in (36, 38, 42):
            ticks = [t for t, _ in self._note_events(out, note)]
            for t, g in zip(ticks, grid):
                assert abs((t - g) - expected) <= 2, (
                    f"note {note}: displacement {t - g}, expected ~{expected}")

    def test_changing_lane_eff_leaves_other_lanes_identical(self, tmp_path):
        """Cross-lane isolation where legitimate — and the honest shared-clock
        behaviour: hats keep driving the kit clock at ANY eff (even 0), so kick
        output is identical whether hats are at 0.9, 0.1, or unset."""
        prof = self._prof()
        inp = self._two_lane_midi(tmp_path, same_tick=False)   # collision-free
        kick_results = []
        hat_results = []
        for i, scales in enumerate([{"hihat_closed": 0.9}, {"hihat_closed": 0.1}, None]):
            o = tmp_path / f"iso{i}.mid"
            humanise(inp, o, prof, intensity=1.0, seed=13, phi=0.5,
                     intensity_by_group=scales)
            kick_results.append(self._note_events(o, 36))
            hat_results.append(self._note_events(o, 42))
        assert kick_results[0] == kick_results[1] == kick_results[2]
        assert hat_results[0] != hat_results[1]

    def test_eff_zero_lane_on_grid(self, tmp_path):
        """A 0.0 lane: solo hits land exactly on their notated ticks with input
        velocities; other lanes are untouched vs the no-dict run."""
        prof = self._prof()
        inp = self._two_lane_midi(tmp_path, same_tick=False)
        out0 = tmp_path / "z0.mid"
        outn = tmp_path / "zn.mid"
        humanise(inp, out0, prof, intensity=1.0, seed=21, phi=0.5,
                 intensity_by_group={"hihat_closed": 0.0})
        humanise(inp, outn, prof, intensity=1.0, seed=21, phi=0.5)
        hat_in = self._note_events(inp, 42)
        assert self._note_events(out0, 42) == hat_in
        assert self._note_events(out0, 36) == self._note_events(outn, 36)

    # -- mode interactions ------------------------------------------------------------

    def test_velocity_only_with_group_dict(self, tmp_path):
        prof = self._prof()
        inp = self._two_lane_midi(tmp_path, same_tick=False)
        out = tmp_path / "vo.mid"
        humanise(inp, out, prof, intensity=1.0, seed=17, phi=0.5,
                 velocity_only=True, intensity_by_group={"kick": 0.0})
        kick_in = self._note_events(inp, 36)
        hat_in = self._note_events(inp, 42)
        kick_out = self._note_events(out, 36)
        hat_out = self._note_events(out, 42)
        # positions untouched everywhere (velocity_only contract)
        assert [t for t, _ in kick_out] == [t for t, _ in kick_in]
        assert [t for t, _ in hat_out] == [t for t, _ in hat_in]
        # kick velocities frozen by eff 0; hat velocities humanised
        assert [v for _, v in kick_out] == [v for _, v in kick_in]
        assert [v for _, v in hat_out] != [v for _, v in hat_in]

    def test_timing_only_with_push_amount(self, tmp_path):
        prof = self._prof(kick_m=50.0, spread=False)
        inp = self._kick_line(tmp_path)
        out = tmp_path / "to.mid"
        humanise(inp, out, prof, intensity=1.0, seed=19, phi=0.4,
                 timing_only=True, push_amount=1.0)
        evs = self._note_events(out, 36)
        assert [v for _, v in evs] == [100] * len(evs)          # velocities untouched
        grid = [4 * self.PPQ * (i + 1) for i in range(8)]
        assert [t for t, _ in evs] != grid                       # timing shifted

    # -- validation ----------------------------------------------------------------------

    def test_validation_errors(self, tmp_path):
        prof = self._prof()
        inp = self._kick_line(tmp_path)
        out = tmp_path / "v.mid"
        with pytest.raises(ValueError, match="unknown instrument group"):
            humanise(inp, out, prof, intensity_by_group={"cowbell": 1.0})
        with pytest.raises(ValueError, match="finite value >= 0"):
            humanise(inp, out, prof, intensity_by_group={"kick": -0.1})
        # nan slips through a bare `v < 0` check (nan < 0 is False); inf would
        # silently blow up tick conversion — both must be rejected explicitly
        with pytest.raises(ValueError, match="finite value >= 0"):
            humanise(inp, out, prof, intensity_by_group={"kick": float("nan")})
        with pytest.raises(ValueError, match="finite value >= 0"):
            humanise(inp, out, prof, intensity_by_group={"kick": float("inf")})
        with pytest.raises(ValueError, match="mutually exclusive"):
            humanise(inp, out, prof, push=True, push_amount=0.5)
        with pytest.raises(ValueError, match="must be in"):
            humanise(inp, out, prof, push_amount=1.5)
        with pytest.raises(ValueError, match="must be in"):
            humanise(inp, out, prof, push_amount=-1.5)


# ---------------------------------------------------------------------------
# TestWindowedCoupling — COUPLE_WINDOW_MS clusters (flams / grace notes)
# Close-spaced ornaments move as ONE rigid unit: shared tick delta from the
# LOUDEST member's sample scaled by the cluster MIN eff, clamped once at
# cluster scope. Same-tick behaviour is regression-locked separately
# (test_pre_round3_fixture_regression — the fixtures predate the window).
# Constant-offset buckets with stored means make every expectation exact:
# the centred sample and residual sigma are both zero, so at push_amount=1.0
# the desired displacement is exactly the bucket mean.
# ---------------------------------------------------------------------------

class TestWindowedCoupling:
    PPQ = 480

    def _prof(self, ms_by_group: dict[str, float]) -> LoadedProfile:
        key = {"kick": "rock|beat|kick", "snare": "rock|beat|snare",
               "hihat_closed": "rock|beat|hihat_closed"}
        buckets = {
            key[g]: BucketProfile(offsets=np.array([m, m, m]),
                                  vel_deltas=np.zeros(3), kde=None)
            for g, m in ms_by_group.items()
        }
        return LoadedProfile(
            buckets=buckets, velocity_thresholds={},
            bucket_offset_means={k: float(b.offsets[0]) for k, b in buckets.items()},
        )

    def _midi(self, tmp_path, notes, tempo_us, note_len=100, name="in.mid") -> Path:
        """Single track; notes = [(tick, midi_note, velocity), ...] sorted."""
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        events = [(t, 1, n, v) for t, n, v in notes]
        events += [(t + note_len, 0, n, 0) for t, n, _v in notes]
        events.sort()
        prev = 0
        for t, kind, n, v in events:
            tr.append(mido.Message("note_on" if kind else "note_off",
                                   channel=9, note=n, velocity=v, time=t - prev))
            prev = t
        tr.append(mido.MetaMessage("end_of_track", time=0))
        p = tmp_path / name
        mid.save(str(p))
        return p

    def _ons(self, path, note=None):
        out = []
        for tr in mido.MidiFile(str(path)).tracks:
            t = 0
            for msg in tr:
                t += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    if note is None or msg.note == note:
                        out.append(t)
        return out

    def test_flam_gap_preserved(self, tmp_path):
        """A 2-note snare flam moves as a unit: internal spacing EXACTLY intact,
        both notes displaced by the same nonzero delta — not collapsed to one
        tick, not scattered."""
        # 120 BPM: 6 ticks = 6.25 ms < 12 ms window. Grace (soft) then main (loud).
        base = 4 * self.PPQ
        inp = self._midi(tmp_path, [(base, 38, 40), (base + 6, 38, 110)],
                         tempo_us=500_000)
        out = tmp_path / "out.mid"
        humanise(inp, out, self._prof({"snare": 40.0}), intensity=1.0, seed=1,
                 phi=0.5, push_amount=1.0)
        a, b = self._ons(out, 38)
        assert b - a == 6, f"flam spacing distorted: {b - a} ticks"
        # loudest (main, on-grid at base+6? no — grid of base+6 is base) drives:
        # cand = quantise(base+6) + 40ms = base + 38 ticks → delta = 32
        assert a - base == 32
        assert b - (base + 6) == 32

    def test_loudest_member_drives_cluster(self, tmp_path):
        """The accented stroke sources the timing, whichever comes first.

        kick bucket mean = +10 ms, snare bucket mean = +40 ms: whoever drives is
        readable straight off the shared displacement."""
        base = 4 * self.PPQ
        prof = self._prof({"kick": 10.0, "snare": 40.0})
        # snare louder → snare's bucket drives: cand = base + 38 → delta = 33
        inp = self._midi(tmp_path, [(base, 36, 60), (base + 5, 38, 120)],
                         tempo_us=500_000, name="snare_loud.mid")
        out = tmp_path / "o1.mid"
        humanise(inp, out, prof, intensity=1.0, seed=1, phi=0.5, push_amount=1.0)
        kick, snare = self._ons(out, 36)[0], self._ons(out, 38)[0]
        assert kick - base == 33 and snare - (base + 5) == 33
        # kick louder → kick's bucket drives: cand = base + 10 → delta = 10
        inp2 = self._midi(tmp_path, [(base, 36, 120), (base + 5, 38, 60)],
                          tempo_us=500_000, name="kick_loud.mid")
        out2 = tmp_path / "o2.mid"
        humanise(inp2, out2, prof, intensity=1.0, seed=1, phi=0.5, push_amount=1.0)
        kick2, snare2 = self._ons(out2, 36)[0], self._ons(out2, 38)[0]
        assert kick2 - base == 10 and snare2 - (base + 5) == 10

    def test_cluster_shares_one_delta_three_members(self, tmp_path):
        base = 4 * self.PPQ
        prof = self._prof({"kick": 50.0, "snare": 50.0, "hihat_closed": 50.0})
        inp = self._midi(tmp_path,
                         [(base, 36, 110), (base + 4, 38, 80), (base + 9, 42, 60)],
                         tempo_us=500_000)
        out = tmp_path / "out.mid"
        humanise(inp, out, prof, intensity=1.0, seed=1, phi=0.5, push_amount=1.0)
        k, s, h = self._ons(out, 36)[0], self._ons(out, 38)[0], self._ons(out, 42)[0]
        deltas = {k - base, s - (base + 4), h - (base + 9)}
        assert len(deltas) == 1, f"members did not share one delta: {deltas}"
        assert s - k == 4 and h - s == 5   # internal spacing intact

    def test_window_boundary_in_ms(self, tmp_path):
        """Inclusive boundary, measured in REAL ms through the tempo map.

        125 BPM, PPQ 480 → 1 tick = exactly 1.0 ms. A 12-tick pair (12.0 ms)
        couples; a 13-tick pair (13.0 ms) does not. Different bucket means make
        coupled vs independent behaviour exactly predictable."""
        base = 4 * self.PPQ
        tempo = 480_000                      # 125 BPM → 1 ms per tick
        prof = self._prof({"kick": 10.0, "snare": 40.0})
        # 12 ms apart → coupled; snare (louder) drives: cand = base+40 → delta 28
        inp = self._midi(tmp_path, [(base, 36, 60), (base + 12, 38, 120)],
                         tempo_us=tempo, name="at12.mid")
        out = tmp_path / "c.mid"
        humanise(inp, out, prof, intensity=1.0, seed=1, phi=0.5, push_amount=1.0)
        kick, snare = self._ons(out, 36)[0], self._ons(out, 38)[0]
        assert kick - base == 28 and snare - (base + 12) == 28
        assert snare - kick == 12            # rigid
        # 13 ms apart → NOT coupled; each lane goes to its own bucket mean
        inp2 = self._midi(tmp_path, [(base, 36, 60), (base + 13, 38, 120)],
                          tempo_us=tempo, name="at13.mid")
        out2 = tmp_path / "u.mid"
        humanise(inp2, out2, prof, intensity=1.0, seed=1, phi=0.5, push_amount=1.0)
        kick2, snare2 = self._ons(out2, 36)[0], self._ons(out2, 38)[0]
        assert kick2 - base == 10            # kick solo → its own +10 ms
        assert snare2 - (base + 13) == 27    # snare solo → grid + 40 → +27 from origin
        assert snare2 - kick2 != 13          # spacing NOT rigidly preserved

    def _offs(self, path, note):
        out = []
        for tr in mido.MidiFile(str(path)).tracks:
            t = 0
            for msg in tr:
                t += msg.time
                is_off = msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0)
                if is_off and getattr(msg, "note", None) == note:
                    out.append(t)
        return out

    def test_cluster_clamp_is_cluster_scoped(self, tmp_path):
        """A member pinned by a GENUINELY FIXED wall (a foreign channel-0 note —
        not shiftable, not a member off) constrains the WHOLE cluster's shared
        delta — flam spacing survives; nobody is clamped independently.

        (A member's OWN note_off is deliberately NOT a wall: member offs are
        elastic and ride with their ons — see the short-note tests below.)
        """
        base = 4 * self.PPQ
        prof = self._prof({"kick": 40.0, "snare": 40.0})
        # kick (loudest) wants +38 ticks; a fixed ch-0 note at +20 caps the
        # snare's musical ceiling at +19 → the CLUSTER moves 14, spacing stays 5.
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
        events = [
            (base, "note_on", 36, 120, 9), (base + 5, "note_on", 38, 90, 9),
            (base + 20, "note_on", 60, 80, 0),
            (base + 300, "note_off", 36, 0, 9), (base + 305, "note_off", 38, 0, 9),
            (base + 320, "note_off", 60, 0, 0),
        ]
        prev = 0
        for t, kind, n, v, ch in events:
            tr.append(mido.Message(kind, channel=ch, note=n, velocity=v, time=t - prev))
            prev = t
        tr.append(mido.MetaMessage("end_of_track", time=0))
        inp = tmp_path / "clamp.mid"
        mid.save(str(inp))

        out = tmp_path / "out.mid"
        humanise(inp, out, prof, intensity=1.0, seed=1, phi=0.5, push_amount=1.0)
        kick, snare = self._ons(out, 36)[0], self._ons(out, 38)[0]
        assert kick - base == 14, f"cluster not clamped as a unit (kick delta {kick - base})"
        assert snare - (base + 5) == 14
        assert snare - kick == 5             # flam spacing survives the clamp
        assert snare == base + 20 - 1        # snare sits at the wall - 1

    def test_short_note_member_cannot_wall_the_cluster(self, tmp_path):
        """Codex P1 repro: PPQ 480, 120 BPM, kick 1920–1955, snare 1950–2050,
        hat 1960–1961 (1 tick — a NORMAL drum note), all buckets +40 ms, phi 0.5.

        Pre-fix: h_lo=4 (kick lands 1954, past the snare's written 1950) and
        h_hi=1 from the hat's own 1-tick LENGTH → hard interval "empty" → the
        per-member fallback smeared snare/hat spacing 10 → 6. A member's own
        duration must never wall the shared delta: its note_off is elastic and
        rides with its on. Expected: uniform +4 nudge, spacing 10.
        """
        prof = self._prof({"kick": 40.0, "snare": 40.0, "hihat_closed": 40.0})
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
        events = [
            (1920, "note_on", 36, 100), (1950, "note_on", 38, 90),
            (1955, "note_off", 36, 0),  (1960, "note_on", 42, 60),
            (1961, "note_off", 42, 0),  (2050, "note_off", 38, 0),
        ]
        prev = 0
        for t, kind, n, v in events:
            tr.append(mido.Message(kind, channel=9, note=n, velocity=v, time=t - prev))
            prev = t
        tr.append(mido.MetaMessage("end_of_track", time=0))
        inp = tmp_path / "codex_repro.mid"
        mid.save(str(inp))

        out = tmp_path / "out.mid"
        humanise(inp, out, prof, intensity=1.0, seed=1, phi=0.5, push_amount=1.0)
        kick = self._ons(out, 36)[0]
        snare = self._ons(out, 38)[0]
        hat = self._ons(out, 42)[0]
        assert kick == 1954                        # prior hit clamped by its own off
        assert hat - snare == 10, f"P1 smear: spacing {hat - snare} (bug gave 6)"
        assert (snare, hat) == (1954, 1964)        # uniform +4, rigid
        # the hat's off rode with its on (encodability without walling the cluster)
        assert self._offs(out, 42)[0] >= hat

    def test_short_member_late_prior_property(self, tmp_path):
        """Property: short-note member (1–2 ticks) + a late prior hit — internal
        spacing is ALWAYS preserved, whole cluster nudged uniformly."""
        for hat_len in (1, 2):
            prof = self._prof({"kick": 40.0, "snare": 40.0, "hihat_closed": 40.0})
            mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
            tr = mido.MidiTrack(); mid.tracks.append(tr)
            tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
            events = sorted([
                (1920, "note_on", 36, 100), (1950, "note_on", 38, 90),
                (1955, "note_off", 36, 0),  (1960, "note_on", 42, 60),
                (1960 + hat_len, "note_off", 42, 0), (2050, "note_off", 38, 0),
            ], key=lambda e: e[0])
            prev = 0
            for t, kind, n, v in events:
                tr.append(mido.Message(kind, channel=9, note=n, velocity=v,
                                       time=t - prev))
                prev = t
            tr.append(mido.MetaMessage("end_of_track", time=0))
            inp = tmp_path / f"prop_{hat_len}.mid"
            mid.save(str(inp))

            out = tmp_path / f"prop_out_{hat_len}.mid"
            humanise(inp, out, prof, intensity=1.0, seed=1, phi=0.5,
                     push_amount=1.0)
            snare = self._ons(out, 38)[0]
            hat = self._ons(out, 42)[0]
            assert hat - snare == 10, (
                f"hat_len={hat_len}: spacing {hat - snare}, expected 10")
            assert snare - 1950 == hat - 1960      # one uniform delta

    def test_empty_intersection_holds_as_unit(self, tmp_path):
        """Empty cluster interval: the flam HOLDS as a unit — never partially
        bumped by per-member emit guards (the pre-fix bug: a 10-tick flam came
        out as 6 because member 1 was pushed by a prior emitted note while
        member 2 held at delta 0).

        Setup forces the empty intersection with BOTH required ingredients:
        a PRIOR EMITTED note landing past member 1's written tick, and a fixed
        note_off BETWEEN the members capping member 1's ceiling below the
        needed lower bound. The only legal rigid move is a uniform +4 nudge.
        """
        prof = self._prof({"kick": 40.0, "snare": 40.0})
        # kick @1920 (len 35 → off @1955) wants +38 but its ceiling (own off)
        # clamps it to 1954 — PAST the first snare's written tick (1950).
        mid = mido.MidiFile(type=0, ticks_per_beat=self.PPQ)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
        events = [
            (1920, "note_on", 36, 100), (1950, "note_on", 38, 90),
            (1955, "note_off", 36, 0),  (1960, "note_on", 38, 110),
            (2050, "note_off", 38, 0),  (2060, "note_off", 38, 0),
        ]
        prev = 0
        for t, kind, n, v in events:
            tr.append(mido.Message(kind, channel=9, note=n, velocity=v, time=t - prev))
            prev = t
        tr.append(mido.MetaMessage("end_of_track", time=0))
        inp = tmp_path / "empty_ix.mid"
        mid.save(str(inp))

        out = tmp_path / "out.mid"
        humanise(inp, out, prof, intensity=1.0, seed=1, phi=0.5, push_amount=1.0)
        kick = self._ons(out, 36)[0]
        s1, s2 = self._ons(out, 38)
        assert kick == 1954                      # prior hit clamped by its own off
        assert s2 - s1 == 10, f"flam smeared: spacing {s2 - s1} (pre-fix bug gave 6)"
        assert (s1, s2) == (1954, 1964)          # whole cluster nudged +4 as a unit

    def test_windowed_cluster_uses_min_eff_of_middle_member(self, tmp_path):
        """gap>0 clusters use cluster_min_eff over ALL members: the min-eff lane
        here is MIDDLE in order and NOT the loudest, so a wrong-lane (loudest /
        first / pairwise first-last) implementation lands elsewhere.

        All buckets mean +50 ms, lean 1: displacement = 50ms × eff. min eff 0.2
        → 10 ms ≈ +1 tick from the loudest member's origin. Loudest-eff (0.9)
        would give +34, first-lane eff (0.6) +20.
        """
        base = 4 * self.PPQ
        prof = self._prof({"kick": 50.0, "snare": 50.0, "hihat_closed": 50.0})
        scales = {"kick": 0.6, "snare": 0.2, "hihat_closed": 0.9}
        inp = self._midi(tmp_path,
                         [(base, 36, 70), (base + 4, 38, 60), (base + 9, 42, 120)],
                         tempo_us=500_000)
        out = tmp_path / "out.mid"
        humanise(inp, out, prof, intensity=1.0, seed=1, phi=0.5,
                 push_amount=1.0, intensity_by_group=scales)
        k = self._ons(out, 36)[0]
        s = self._ons(out, 38)[0]
        h = self._ons(out, 42)[0]
        # loudest (hat @base+9): cand = grid + round(50*0.2 ms) = base+10 → delta +1
        assert (k - base, s - (base + 4), h - (base + 9)) == (1, 1, 1), (
            f"cluster did not use min eff: deltas "
            f"{(k - base, s - (base + 4), h - (base + 9))}")

    def test_eager_draw_order_and_stash_consumption(self, tmp_path):
        """Sample-consumption order across adjacent windowed clusters and a
        same-tick chord, locked via tagged velocities.

        Buckets carry distinct per-draw vel_delta tags; the expected stream is
        re-simulated with the same seeded RNG (one randint per shiftable hit in
        merged order — the engine contract). A mis-keyed stash or out-of-order
        eager draw assigns some hit another draw's tag and fails.
        """
        snare_tags = [11.0, 12.0, 13.0, 14.0, 15.0]
        hat_tags = [21.0, 22.0, 23.0, 24.0, 25.0]
        prof = LoadedProfile(
            buckets={
                "rock|beat|snare": BucketProfile(
                    offsets=np.zeros(5), vel_deltas=np.array(snare_tags), kde=None),
                "rock|beat|hihat_closed": BucketProfile(
                    offsets=np.zeros(5), vel_deltas=np.array(hat_tags), kde=None),
            },
            velocity_thresholds={},
            bucket_offset_means={"rock|beat|snare": 0.0,
                                 "rock|beat|hihat_closed": 0.0},
        )
        # clusters A and B (6-tick flams), then a SAME-TICK chord C: the chord
        # draws through the normal (non-stash) path, so stream continuity across
        # the stash boundary is exercised too.
        notes = [
            (1920, 38, 80), (1926, 42, 60),     # cluster A (windowed)
            (2400, 38, 80), (2406, 42, 60),     # cluster B (windowed)
            (2880, 38, 80), (2880, 42, 60),     # chord C (same-tick, legacy path)
        ]
        inp = self._midi(tmp_path, notes, tempo_us=500_000, note_len=60)
        out = tmp_path / "out.mid"
        humanise(inp, out, prof, intensity=1.0, seed=7, phi=0.5)

        # expected: one uniform-index draw per shiftable hit, in merged order
        np.random.seed(7)
        draw_idx = [int(np.random.randint(5)) for _ in range(6)]
        lane_tags = [snare_tags, hat_tags, snare_tags, hat_tags, snare_tags, hat_tags]
        bases = [80, 60, 80, 60, 80, 60]
        expected = [round(b + tags[i]) for b, tags, i in
                    zip(bases, lane_tags, draw_idx)]

        got = []
        for track in mido.MidiFile(str(out)).tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    got.append(msg.velocity)
        assert got == expected, f"velocities {got} != expected tags {expected}"
