"""Tests for wobblemidi_gui/adapter.py — the UI-free engine adapter.

Uses the real bundled rock.json profile (loaded once per module) so profile
resolution from the installed package is exercised too. No pywebview needed:
the adapter must stay importable without any UI framework installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mido
import pytest

from wobblemidi_gui import adapter
from wobblemidi_gui.adapter import LANE_ORDER, Session, parse_for_display

DEFAULT_PPQ = 480
DEFAULT_TEMPO_US = 500_000  # 120 BPM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_midi(
    messages: list[mido.Message],
    ppq: int = DEFAULT_PPQ,
    tempo_us: int = DEFAULT_TEMPO_US,
    midi_type: int = 0,
    time_sig: tuple[int, int] | None = None,
) -> mido.MidiFile:
    mid = mido.MidiFile(type=midi_type, ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    if time_sig is not None:
        track.append(mido.MetaMessage(
            "time_signature", numerator=time_sig[0], denominator=time_sig[1], time=0
        ))
    for msg in messages:
        track.append(msg)
    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def _on(note: int, vel: int = 100, time: int = 0, channel: int = 9) -> mido.Message:
    return mido.Message("note_on", note=note, velocity=vel, time=time, channel=channel)


def _off(note: int, time: int = 0, channel: int = 9) -> mido.Message:
    return mido.Message("note_off", note=note, velocity=0, time=time, channel=channel)


NOTE_LEN = 60  # ticks — real note lengths; zero-length notes make the engine's
               # windowing clamp a hit below its own same-tick note_off.


def _two_bar_beat(tmp_path: Path) -> Path:
    """Two 4/4 bars, on-grid: kick on 1 & 3, snare on 2 & 4, 8th closed hats."""
    q = DEFAULT_PPQ
    ons = []
    for bar in range(2):
        base = bar * 4 * q
        ons += [(base, 36, 110), (base + 2 * q, 36, 105)]
        ons += [(base + q, 38, 95), (base + 3 * q, 38, 100)]
        ons += [(base + e * q // 2, 42, 70 + (e % 2) * 20) for e in range(8)]
    events = [(tick, 1, note, vel) for tick, note, vel in ons]
    events += [(tick + NOTE_LEN, 0, note, 0) for tick, note, _vel in ons]
    events.sort()
    msgs: list[mido.Message] = []
    prev = 0
    for tick, kind, note, vel in events:
        if kind == 1:
            msgs.append(_on(note, vel, time=tick - prev))
        else:
            msgs.append(_off(note, time=tick - prev))
        prev = tick
    p = tmp_path / "two_bar.mid"
    _make_midi(msgs, time_sig=(4, 4)).save(str(p))
    return p


@pytest.fixture(scope="module")
def session() -> Session:
    s = Session()
    yield s
    s.cleanup()


# ---------------------------------------------------------------------------
# parse_for_display
# ---------------------------------------------------------------------------

class TestParseForDisplay:
    def test_basic_parse(self, tmp_path):
        p = _two_bar_beat(tmp_path)
        d = parse_for_display(p)
        assert d["lanes"] == LANE_ORDER
        assert len(d["hits"]) == 2 * (2 + 2 + 8)
        lanes = {h["lane"] for h in d["hits"]}
        assert lanes == {"kick", "snare", "hihat_closed"}
        # tick 480 @ 120 BPM, ppq 480 → 500 ms
        snare_1 = [h for h in d["hits"] if h["lane"] == "snare"][0]
        assert snare_1["tick"] == 480
        assert snare_1["ms"] == pytest.approx(500.0)
        assert d["bpm"] == pytest.approx(120.0)

    def test_bars_and_beats_four_four(self, tmp_path):
        p = _two_bar_beat(tmp_path)
        d = parse_for_display(p)
        assert [b["tick"] for b in d["bars"]] == [0, 4 * DEFAULT_PPQ]
        assert d["bars"][1]["ms"] == pytest.approx(2000.0)
        # 4 beats per bar
        assert len(d["beats"]) == 8

    def test_bars_three_four(self, tmp_path):
        bar = 3 * DEFAULT_PPQ
        msgs = [_on(36, 100, time=0), _off(36)]
        for _ in range(2):
            msgs += [_on(36, 100, time=bar), _off(36)]
        p = tmp_path / "waltz.mid"
        _make_midi(msgs, time_sig=(3, 4)).save(str(p))
        d = parse_for_display(p)
        assert [b["tick"] for b in d["bars"]] == [0, bar]
        assert d["bars"][1]["tick"] - d["bars"][0]["tick"] == bar

    def test_melodic_channel_excluded_by_default(self, tmp_path):
        msgs = [
            _on(36, 100, time=0), _off(36),
            _on(36, 100, time=480, channel=0), _off(36, channel=0),
        ]
        p = tmp_path / "mixed.mid"
        _make_midi(msgs).save(str(p))
        d = parse_for_display(p)
        assert len(d["hits"]) == 1
        assert d["other_channel_drum_hits"] == 1
        d_all = parse_for_display(p, all_channels=True)
        assert len(d_all["hits"]) == 2
        assert d_all["other_channel_drum_hits"] == 0

    def test_type2_rejected(self, tmp_path):
        p = tmp_path / "t2.mid"
        _make_midi([_on(36, 100), _off(36)], midi_type=2).save(str(p))
        with pytest.raises(ValueError, match="Type 2"):
            parse_for_display(p)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class TestSessionLoad:
    def test_load_ok(self, session, tmp_path):
        res = session.load(_two_bar_beat(tmp_path))
        assert res["ok"] is True
        assert res["file_name"] == "two_bar.mid"
        assert res["warnings"] == []
        assert len(res["original"]["hits"]) == 24

    def test_load_type2_error_dict(self, session, tmp_path):
        p = tmp_path / "t2.mid"
        _make_midi([_on(36, 100), _off(36)], midi_type=2).save(str(p))
        res = session.load(p)
        assert res["ok"] is False
        assert "Type 2" in res["error"]

    def test_load_mixed_six_eight_error_dict(self, session, tmp_path):
        mid = mido.MidiFile(type=0, ticks_per_beat=DEFAULT_PPQ)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("time_signature", numerator=6, denominator=8, time=0))
        track.append(_on(36, 100, time=0))
        track.append(_off(36))
        track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=960))
        track.append(mido.MetaMessage("end_of_track", time=0))
        p = tmp_path / "mixed_meter.mid"
        mid.save(str(p))
        res = session.load(p)
        assert res["ok"] is False
        assert "Mixed time signatures" in res["error"]

    def test_load_corrupt_error_dict(self, session, tmp_path):
        p = tmp_path / "junk.mid"
        p.write_bytes(b"this is not midi")
        res = session.load(p)
        assert res["ok"] is False

    def test_no_drum_hits_warning(self, session, tmp_path):
        msgs = [_on(36, 100, channel=0), _off(36, channel=0)]
        p = tmp_path / "melodic.mid"
        _make_midi(msgs).save(str(p))
        res = session.load(p)
        assert res["ok"] is True
        assert res["warnings"] == ["no_drum_hits_channel10"]

    def test_humanise_without_load(self):
        s = Session.__new__(Session)  # skip profile load; original_path check first
        s.original_path = None
        res = Session.humanise_current(s, {})
        assert res["ok"] is False
        assert "No MIDI file loaded" in res["error"]


PARAMS = {"intensity": 0.35, "tightness": 0.4, "lean": 0.0, "all_channels": False,
          "lane_intensity": {}}


class TestSessionHumanise:
    def test_humanise_flow(self, session, tmp_path):
        session.load(_two_bar_beat(tmp_path))
        res = session.humanise_current(dict(PARAMS))
        assert res["ok"] is True
        assert isinstance(res["seed"], int)
        assert res["can_undo"] is False       # first render — nothing to undo
        assert session.render_path is not None and session.render_path.exists()
        orig, new = res["original"], res["humanised"]
        assert len(new["hits"]) == len(orig["hits"])
        # deltas attached and reference the original position
        for h in new["hits"]:
            assert h["ms"] == pytest.approx(h["orig_ms"] + h["delta_ms"])

    def test_every_render_is_a_new_take(self, session, tmp_path):
        session.load(_two_bar_beat(tmp_path))
        r1 = session.humanise_current(dict(PARAMS))
        r2 = session.humanise_current(dict(PARAMS))
        assert r1["seed"] != r2["seed"]

    def test_lane_intensity_passthrough(self, session, tmp_path):
        session.load(_two_bar_beat(tmp_path))
        res = session.humanise_current(
            {**PARAMS, "intensity": 1.0, "lane_intensity": {"kick": 0.0}})
        assert res["ok"] is True
        kick = [h for h in res["humanised"]["hits"] if h["lane"] == "kick"]
        other = [h for h in res["humanised"]["hits"] if h["lane"] != "kick"]
        assert kick and all(h["delta_ms"] == 0 and h["delta_vel"] == 0 for h in kick)
        assert any(h["delta_ms"] != 0 for h in other)

    def test_engine_validation_surfaces_as_error(self, session, tmp_path):
        session.load(_two_bar_beat(tmp_path))
        res = session.humanise_current({**PARAMS, "lean": 2.0})
        assert res["ok"] is False
        assert "push_amount" in res["error"]

    def test_rehumanise_always_from_original(self, session, tmp_path):
        """Render 2 at intensity 0 must land on the ORIGINAL grid — proving it
        was derived from the loaded file, not from render 1 (preserve-intent)."""
        p = _two_bar_beat(tmp_path)
        session.load(p)
        r1 = session.humanise_current(dict(PARAMS))
        assert any(h["delta_ms"] != 0 for h in r1["humanised"]["hits"])
        r2 = session.humanise_current({**PARAMS, "intensity": 0.0})
        for h in r2["humanised"]["hits"]:
            assert h["delta_ms"] == pytest.approx(0.0)
            assert h["delta_vel"] == 0

    def test_new_file_resets_seed(self, session, tmp_path):
        session.load(_two_bar_beat(tmp_path))
        session.humanise_current(dict(PARAMS))
        assert session.seed is not None
        session.load(_two_bar_beat(tmp_path))
        assert session.seed is None
        assert session.render_path is None
        assert session.undo()["ok"] is False   # undo history cleared with the file


class TestSessionUndo:
    def test_undo_before_second_render(self, session, tmp_path):
        session.load(_two_bar_beat(tmp_path))
        assert session.undo()["ok"] is False
        session.humanise_current(dict(PARAMS))
        assert session.undo()["ok"] is False   # single render — no previous yet

    def test_undo_swaps_and_redoes(self, session, tmp_path):
        session.load(_two_bar_beat(tmp_path))
        r1 = session.humanise_current({**PARAMS, "intensity": 0.35})
        r2 = session.humanise_current({**PARAMS, "intensity": 0.9})
        assert r2["can_undo"] is True

        u1 = session.undo()
        assert u1["ok"] is True
        assert u1["seed"] == r1["seed"]
        assert u1["params"]["intensity"] == pytest.approx(0.35)
        assert [h["ms"] for h in u1["humanised"]["hits"]] == \
               [h["ms"] for h in r1["humanised"]["hits"]]

        u2 = session.undo()   # undo twice = redo
        assert u2["seed"] == r2["seed"]
        assert u2["params"]["intensity"] == pytest.approx(0.9)
        assert [h["ms"] for h in u2["humanised"]["hits"]] == \
               [h["ms"] for h in r2["humanised"]["hits"]]

    def test_export_follows_undo(self, session, tmp_path):
        session.load(_two_bar_beat(tmp_path))
        session.humanise_current(dict(PARAMS))
        first_render = session.render_path
        session.humanise_current({**PARAMS, "intensity": 0.9})
        session.undo()
        dest = tmp_path / "undone.mid"
        assert session.export_to(dest)["ok"] is True
        assert dest.read_bytes() == first_render.read_bytes()


class TestSessionExport:
    def test_export_before_render(self, session, tmp_path):
        session.load(_two_bar_beat(tmp_path))
        res = session.export_to(tmp_path / "out.mid")
        assert res["ok"] is False

    def test_export_copies_render(self, session, tmp_path):
        session.load(_two_bar_beat(tmp_path))
        session.humanise_current(dict(PARAMS))
        dest = tmp_path / "exported.mid"
        res = session.export_to(dest)
        assert res["ok"] is True
        assert dest.read_bytes() == session.render_path.read_bytes()


class TestSeam:
    def test_adapter_is_pywebview_free(self):
        """adapter.py is the future web-demo seam — importing it must not pull
        in pywebview (only app.py may)."""
        assert "wobblemidi_gui.adapter" in sys.modules
        assert "webview" not in sys.modules


def test_cleanup_removes_tmpdir(tmp_path):
    s = Session()
    s.load(_two_bar_beat(tmp_path))
    s.humanise_current(dict(PARAMS))
    d = s._tmpdir
    assert d.exists()
    s.cleanup()
    assert not d.exists()
