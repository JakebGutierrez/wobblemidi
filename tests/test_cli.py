"""Tests for pocketmidi/cli.py"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch, call

import mido
import numpy as np
import pytest
from click.testing import CliRunner

from pocketmidi.cli import main
from pocketmidi.humanise import BucketProfile, LoadedProfile

# ---------------------------------------------------------------------------
# Helpers (minimal copies of test_humanise.py utilities)
# ---------------------------------------------------------------------------

DEFAULT_PPQ = 480
DEFAULT_TEMPO_US = 500_000  # 120 BPM


def _make_midi(
    messages: list[mido.Message],
    ppq: int = DEFAULT_PPQ,
    tempo_us: int = DEFAULT_TEMPO_US,
    midi_type: int = 0,
) -> mido.MidiFile:
    mid = mido.MidiFile(type=midi_type, ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    for msg in messages:
        track.append(msg)
    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def _minimal_midi(tmp_path: Path) -> Path:
    """Write a minimal one-hit MIDI file and return its path."""
    mid = _make_midi([
        mido.Message("note_on",  channel=9, note=36, velocity=80, time=0),
        mido.Message("note_off", channel=9, note=36, velocity=0,  time=480),
    ])
    p = tmp_path / "input.mid"
    mid.save(str(p))
    return p


_FAKE_PROFILES = LoadedProfile(
    buckets={
        "rock|beat|kick": BucketProfile(
            offsets=np.array([0.0, 5.0, -5.0]),
            vel_deltas=np.array([0.0, 10.0, -10.0]),
            kde=None,  # degenerate-safe; tests don't inspect KDE output
        ),
    },
    velocity_thresholds={},
)

_SENTINEL = Path("/nonexistent/rock.json")


@contextmanager
def _fake_as_file(traversable):
    """Bypass real resource lookup; load_profile is always mocked alongside this."""
    yield _SENTINEL


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_output_file_created(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES):
            result = runner.invoke(main, [str(in_path), str(out_path)])

        assert result.exit_code == 0, result.output
        assert out_path.exists()

    def test_load_profile_called_with_resolved_path(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES) as mock_lp:
            runner.invoke(main, [str(in_path), str(out_path)])

        mock_lp.assert_called_once_with(_SENTINEL)


class TestErrorHandling:
    def test_missing_genre_profile(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        @contextmanager
        def raise_fnf(traversable):
            raise FileNotFoundError
            yield  # pragma: no cover

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", raise_fnf):
            result = runner.invoke(main, [str(in_path), str(out_path), "--genre", "jazz"])

        assert result.exit_code == 1
        assert "no profile found for genre 'jazz'" in result.output

    def test_midi_type2_raises_value_error(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES), \
             patch("pocketmidi.cli.humanise", side_effect=ValueError("MIDI type 2 not supported")):
            result = runner.invoke(main, [str(in_path), str(out_path)])

        assert result.exit_code == 1
        assert "MIDI type 2 not supported" in result.output

    def test_intensity_out_of_range(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        result = runner.invoke(main, [str(in_path), str(out_path), "--intensity", "1.5"])

        assert result.exit_code != 0


class TestFlags:
    def test_seed_determinism(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out1 = tmp_path / "out1.mid"
        out2 = tmp_path / "out2.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES):
            runner.invoke(main, [str(in_path), str(out1), "--seed", "42"])
            runner.invoke(main, [str(in_path), str(out2), "--seed", "42"])

        assert out1.read_bytes() == out2.read_bytes()

    def test_section_flag_passed_through(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES), \
             patch("pocketmidi.cli.humanise") as mock_h:
            runner.invoke(main, [str(in_path), str(out_path), "--section", "fill"])

        _, kwargs = mock_h.call_args
        assert kwargs["beat_type"] == "fill"

    def test_intensity_flag_passed_through(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES), \
             patch("pocketmidi.cli.humanise") as mock_h:
            runner.invoke(main, [str(in_path), str(out_path), "--intensity", "0.5"])

        _, kwargs = mock_h.call_args
        assert kwargs["intensity"] == pytest.approx(0.5)

    def test_verbose_no_crash(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES):
            result = runner.invoke(main, [str(in_path), str(out_path), "--verbose"])

        assert result.exit_code == 0

    def test_timing_only_passed_through(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES), \
             patch("pocketmidi.cli.humanise") as mock_h:
            runner.invoke(main, [str(in_path), str(out_path), "--timing-only"])

        _, kwargs = mock_h.call_args
        assert kwargs["timing_only"] is True
        assert kwargs["velocity_only"] is False

    def test_velocity_only_passed_through(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES), \
             patch("pocketmidi.cli.humanise") as mock_h:
            runner.invoke(main, [str(in_path), str(out_path), "--velocity-only"])

        _, kwargs = mock_h.call_args
        assert kwargs["velocity_only"] is True
        assert kwargs["timing_only"] is False

    def test_timing_and_velocity_only_exclusive(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        result = runner.invoke(main, [str(in_path), str(out_path),
                                      "--timing-only", "--velocity-only"])

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_push_passed_through(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES), \
             patch("pocketmidi.cli.humanise") as mock_h:
            runner.invoke(main, [str(in_path), str(out_path), "--push"])

        _, kwargs = mock_h.call_args
        assert kwargs["push"] is True

    def test_push_default_false(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES), \
             patch("pocketmidi.cli.humanise") as mock_h:
            runner.invoke(main, [str(in_path), str(out_path)])

        _, kwargs = mock_h.call_args
        assert kwargs["push"] is False

    def test_groove_tightness_passed_through(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES), \
             patch("pocketmidi.cli.humanise") as mock_h:
            runner.invoke(main, [str(in_path), str(out_path), "--groove-tightness", "0.7"])

        _, kwargs = mock_h.call_args
        assert kwargs["phi"] == pytest.approx(0.7)

    def test_groove_tightness_default_is_0_4(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES), \
             patch("pocketmidi.cli.humanise") as mock_h:
            runner.invoke(main, [str(in_path), str(out_path)])

        _, kwargs = mock_h.call_args
        assert kwargs["phi"] == pytest.approx(0.4)

    def test_all_channels_passed_through(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES), \
             patch("pocketmidi.cli.humanise") as mock_h:
            runner.invoke(main, [str(in_path), str(out_path), "--all-channels"])

        _, kwargs = mock_h.call_args
        assert kwargs["all_channels"] is True

    def test_all_channels_default_false(self, tmp_path):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        with patch("pocketmidi.cli.as_file", _fake_as_file), \
             patch("pocketmidi.cli.load_profile", return_value=_FAKE_PROFILES), \
             patch("pocketmidi.cli.humanise") as mock_h:
            runner.invoke(main, [str(in_path), str(out_path)])

        _, kwargs = mock_h.call_args
        assert kwargs["all_channels"] is False

    @pytest.mark.parametrize("bad", ["1.0", "-0.1"])
    def test_groove_tightness_out_of_range(self, tmp_path, bad):
        in_path = _minimal_midi(tmp_path)
        out_path = tmp_path / "out.mid"

        runner = CliRunner()
        result = runner.invoke(main, [str(in_path), str(out_path), "--groove-tightness", bad])

        assert result.exit_code != 0
