"""Tests for pocketmidi.midi_utils."""

import pytest
import mido

from pocketmidi.midi_utils import (
    TD11_TO_GROUP,
    build_tempo_map,
    get_tempo_at_tick,
    grid_position_in_bar,
    is_four_four,
    offset_ticks_to_ms,
    quantise_to_grid,
    ticks_to_ms,
    ticks_to_ms_with_map,
)


# ---------------------------------------------------------------------------
# TD11_TO_GROUP mapping
# ---------------------------------------------------------------------------

class TestTD11ToGroup:
    def test_kick(self):
        assert TD11_TO_GROUP[36] == "kick"

    def test_snare_variants(self):
        for note in (38, 40, 37):
            assert TD11_TO_GROUP[note] == "snare"

    def test_hihat_closed_edge(self):
        # Note 22 is HH Closed (Edge) per Magenta GMD mapping table
        assert TD11_TO_GROUP[22] == "hihat_closed"

    def test_hihat_open_edge(self):
        # Note 26 is HH Open (Edge) per Magenta GMD mapping table
        assert TD11_TO_GROUP[26] == "hihat_open"

    def test_crash_variants(self):
        for note in (49, 55, 57, 52):
            assert TD11_TO_GROUP[note] == "crash"

    def test_ride_variants(self):
        for note in (51, 59, 53):
            assert TD11_TO_GROUP[note] == "ride"

    def test_tom_groups(self):
        assert TD11_TO_GROUP[48] == "tom_high"
        assert TD11_TO_GROUP[45] == "tom_mid"
        assert TD11_TO_GROUP[43] == "tom_low"

    def test_all_21_notes_mapped(self):
        assert len(TD11_TO_GROUP) == 22  # 21 physical + note 22 (added in spec)

    def test_unknown_note_not_present(self):
        assert 0 not in TD11_TO_GROUP
        assert 127 not in TD11_TO_GROUP


# ---------------------------------------------------------------------------
# ticks_to_ms
# ---------------------------------------------------------------------------

class TestTicksToMs:
    def test_120bpm_one_beat(self):
        # 480 ticks at 120 BPM (500_000 µs/beat, ppq=480) = 500 ms
        assert ticks_to_ms(480, 500_000, 480) == pytest.approx(500.0)

    def test_120bpm_one_sixteenth(self):
        # 120 ticks at 120 BPM = 125 ms
        assert ticks_to_ms(120, 500_000, 480) == pytest.approx(125.0)

    def test_zero_ticks(self):
        assert ticks_to_ms(0, 500_000, 480) == pytest.approx(0.0)

    def test_invalid_ppq(self):
        with pytest.raises(ValueError):
            ticks_to_ms(480, 500_000, 0)

    def test_different_tempo(self):
        # 240 BPM = 250_000 µs/beat; 480 ticks = 250 ms
        assert ticks_to_ms(480, 250_000, 480) == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# quantise_to_grid
# ---------------------------------------------------------------------------

class TestQuantiseToGrid:
    # ppq=480 → 16th note = 120 ticks

    def test_on_grid(self):
        assert quantise_to_grid(480, 480) == 480  # exactly on beat 2

    def test_slightly_late_snaps_back(self):
        # 482 ticks: 2 ticks after grid at 480 → snap to 480
        assert quantise_to_grid(482, 480) == 480

    def test_slightly_early_snaps_back(self):
        # 478 ticks: 2 ticks before grid at 480 → snap to 480
        assert quantise_to_grid(478, 480) == 480

    def test_halfway_snaps_forward(self):
        # 480 + 60 = 540: exactly halfway → snap forward to 600
        assert quantise_to_grid(540, 480) == 600

    def test_first_sixteenth(self):
        assert quantise_to_grid(0, 480) == 0

    def test_invalid_ppq(self):
        with pytest.raises(ValueError):
            quantise_to_grid(480, 0)

    def test_slightly_before_next_grid(self):
        # 119 ticks with ppq=480: sixteenth=120; 119 < 60 so snaps back to 0
        assert quantise_to_grid(119, 480) == 120

    def test_ppq_220(self):
        # ppq=220 → sixteenth=55; note at 57 → 2 past grid at 55 → snap to 55
        assert quantise_to_grid(57, 220) == 55


# ---------------------------------------------------------------------------
# offset_ticks_to_ms
# ---------------------------------------------------------------------------

class TestOffsetTicksToMs:
    def test_positive_offset_is_late(self):
        # Note at tick 482, grid at 480, 2 ticks late at 120 BPM
        offset = offset_ticks_to_ms(482, 480, 500_000, 480)
        assert offset > 0

    def test_negative_offset_is_early(self):
        offset = offset_ticks_to_ms(478, 480, 500_000, 480)
        assert offset < 0

    def test_on_grid_is_zero(self):
        assert offset_ticks_to_ms(480, 480, 500_000, 480) == pytest.approx(0.0)

    def test_magnitude(self):
        # 120 ticks (1 sixteenth) late at 120 BPM = 125 ms
        assert offset_ticks_to_ms(600, 480, 500_000, 480) == pytest.approx(125.0)


# ---------------------------------------------------------------------------
# get_tempo_at_tick
# ---------------------------------------------------------------------------

class TestGetTempoAtTick:
    TEMPO_MAP = [(0, 500_000), (1920, 400_000), (3840, 600_000)]

    def test_first_tempo(self):
        assert get_tempo_at_tick(self.TEMPO_MAP, 0) == 500_000

    def test_mid_first_segment(self):
        assert get_tempo_at_tick(self.TEMPO_MAP, 960) == 500_000

    def test_at_second_event(self):
        assert get_tempo_at_tick(self.TEMPO_MAP, 1920) == 400_000

    def test_mid_second_segment(self):
        assert get_tempo_at_tick(self.TEMPO_MAP, 2500) == 400_000

    def test_at_third_event(self):
        assert get_tempo_at_tick(self.TEMPO_MAP, 3840) == 600_000

    def test_beyond_last_event(self):
        assert get_tempo_at_tick(self.TEMPO_MAP, 9999) == 600_000


# ---------------------------------------------------------------------------
# ticks_to_ms_with_map
# ---------------------------------------------------------------------------

class TestTicksToMsWithMap:
    # ppq=480, one tempo throughout: 120 BPM (500_000 µs/beat)
    FLAT_MAP = [(0, 500_000)]

    # Tempo changes at tick 1920 (beat 4): 120 BPM → 150 BPM (400_000 µs/beat)
    CHANGE_MAP = [(0, 500_000), (1920, 400_000)]

    def test_constant_tempo_one_beat(self):
        # 480 ticks at 120 BPM = 500 ms — same result as ticks_to_ms
        assert ticks_to_ms_with_map(0, 480, self.FLAT_MAP, 480) == pytest.approx(500.0)

    def test_constant_tempo_zero_range(self):
        assert ticks_to_ms_with_map(100, 100, self.FLAT_MAP, 480) == pytest.approx(0.0)

    def test_range_entirely_before_tempo_change(self):
        # 0–960 ticks, all at 500_000 µs/beat → 1000 ms
        assert ticks_to_ms_with_map(0, 960, self.CHANGE_MAP, 480) == pytest.approx(1000.0)

    def test_range_entirely_after_tempo_change(self):
        # 1920–2400 ticks (480 ticks), all at 400_000 µs/beat → 400 ms
        assert ticks_to_ms_with_map(1920, 2400, self.CHANGE_MAP, 480) == pytest.approx(400.0)

    def test_range_spanning_tempo_change(self):
        # 960–2400: first 960 ticks at 500_000 → 1000 ms,
        #           then 480 ticks at 400_000 → 400 ms; total = 1400 ms
        assert ticks_to_ms_with_map(960, 2400, self.CHANGE_MAP, 480) == pytest.approx(1400.0)

    def test_invalid_ppq(self):
        with pytest.raises(ValueError):
            ticks_to_ms_with_map(0, 480, self.FLAT_MAP, 0)


# ---------------------------------------------------------------------------
# TestGridPositionInBar
# ---------------------------------------------------------------------------

class TestGridPositionInBar:
    PPQ = 480  # sixteenth = 120 ticks, bar = 1920 ticks

    def test_beat1_downbeat(self):
        assert grid_position_in_bar(0, self.PPQ) == 0

    def test_beat2_downbeat(self):
        # beat 2 = 4 sixteenths in = tick 480
        assert grid_position_in_bar(480, self.PPQ) == 4

    def test_beat3_downbeat(self):
        assert grid_position_in_bar(960, self.PPQ) == 8

    def test_beat4_downbeat(self):
        assert grid_position_in_bar(1440, self.PPQ) == 12

    def test_last_sixteenth(self):
        # position 15 = tick 1800 (15 * 120)
        assert grid_position_in_bar(1800, self.PPQ) == 15

    def test_wraps_at_bar_boundary(self):
        # tick 1920 = start of bar 2 → position 0
        assert grid_position_in_bar(1920, self.PPQ) == 0

    def test_second_bar_beat2(self):
        # bar 2, beat 2 = tick 1920 + 480 = 2400 → position 4
        assert grid_position_in_bar(2400, self.PPQ) == 4

    def test_non_standard_ppq(self):
        ppq = 220  # sixteenth = 55 ticks, bar = 880 ticks
        assert grid_position_in_bar(0, ppq) == 0
        assert grid_position_in_bar(55, ppq) == 1
        assert grid_position_in_bar(220, ppq) == 4

    def test_non_divisible_ppq_wraps_correctly(self):
        # ppq=222: sixteenth = 222 // 4 = 55 (truncates), ticks_per_bar = 16 * 55 = 880.
        # Using ppq * 4 = 888 instead would make position 15 (tick 825) + one step
        # land at tick 880 which gives (880 % 888) // 55 = 16 — out of range.
        ppq = 222
        sixteenth = 55  # 222 // 4
        assert grid_position_in_bar(0, ppq) == 0
        assert grid_position_in_bar(15 * sixteenth, ppq) == 15   # last slot
        assert grid_position_in_bar(16 * sixteenth, ppq) == 0    # wraps to next bar


# ---------------------------------------------------------------------------
# TestIsFourFour
# ---------------------------------------------------------------------------

def _midi_with_time_sig(numerator: int, denominator: int) -> mido.MidiFile:
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage(
        "time_signature",
        numerator=numerator,
        denominator=denominator,
        clocks_per_click=24,
        notated_32nd_notes_per_beat=8,
        time=0,
    ))
    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


class TestIsFourFour:
    def test_no_time_sig_is_four_four(self):
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("end_of_track", time=0))
        assert is_four_four(mid) is True

    def test_explicit_four_four(self):
        assert is_four_four(_midi_with_time_sig(4, 4)) is True

    def test_three_four(self):
        assert is_four_four(_midi_with_time_sig(3, 4)) is False

    def test_six_eight(self):
        assert is_four_four(_midi_with_time_sig(6, 8)) is False

    def test_five_four(self):
        assert is_four_four(_midi_with_time_sig(5, 4)) is False
