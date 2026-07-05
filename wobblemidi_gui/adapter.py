"""Engine adapter for the wobblemidi GUI — UI-framework-free.

This module is the seam between any front-end shell and the wobblemidi engine.
It must stay free of pywebview (and any other UI) imports: a future thin
web-demo server should be able to wrap Session unchanged. app.py owns windows
and dialogs; this module owns files, state, and engine calls.

All Session methods return plain JSON-serialisable dicts shaped
``{"ok": True, ...}`` or ``{"ok": False, "error": "<user-facing message>"}`` —
they never raise across the bridge.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import threading
from collections import defaultdict
from importlib.resources import as_file, files
from pathlib import Path

import mido

from wobblemidi.humanise import DRUM_CHANNEL, humanise, load_profile
from wobblemidi.midi_utils import (
    TD11_TO_GROUP,
    build_tempo_map,
    detect_meter,
    ticks_to_ms_with_map,
)

# Screen lane order, top to bottom — drum-editor convention: cymbals up top,
# kick on the floor.
LANE_ORDER = [
    "crash",
    "ride",
    "hihat_open",
    "hihat_closed",
    "tom_high",
    "tom_mid",
    "tom_low",
    "snare",
    "kick",
]


def _collapse_time_sigs(sigs: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Sort (tick, num, den) events; same-tick duplicates keep the last one."""
    out: dict[int, tuple[int, int, int]] = {}
    for tick, num, den in sorted(sigs, key=lambda s: s[0]):
        out[tick] = (tick, num, den)
    return [out[t] for t in sorted(out)]


def _bars_and_beats(
    time_sigs: list[tuple[int, int, int]],
    end_tick: int,
    ppq: int,
    tempo_map: list[tuple[int, int]],
) -> tuple[list[dict], list[dict]]:
    """Bar and beat gridlines as {tick, ms}, honouring time-signature changes.

    A signature event restarts the bar count at its own tick (standard notation
    behaviour). ms values go through the tempo map, so gridlines stay correct
    under tempo changes.
    """
    sigs = _collapse_time_sigs(time_sigs)
    if not sigs or sigs[0][0] != 0:
        sigs.insert(0, (0, 4, 4))  # MIDI default meter

    bars: list[dict] = []
    beats: list[dict] = []
    for i, (seg_start, num, den) in enumerate(sigs):
        seg_end = sigs[i + 1][0] if i + 1 < len(sigs) else end_tick
        bar_len = max(1, (num * 4 * ppq) // den)
        beat_len = max(1, bar_len // num)
        t = seg_start
        while t < seg_end:
            bars.append({"tick": t, "ms": ticks_to_ms_with_map(0, t, tempo_map, ppq)})
            b = t
            bar_end = min(t + bar_len, seg_end)
            while b < bar_end:
                beats.append({"tick": b, "ms": ticks_to_ms_with_map(0, b, tempo_map, ppq)})
                b += beat_len
            t += bar_len
    return bars, beats


def parse_for_display(path: str | Path, all_channels: bool = False) -> dict:
    """Parse a MIDI file into the display/playback structure the front-end draws.

    Hits are emitted in track-major message order (the same order the engine
    preserves), so two parses of an input and its humanised output pair up
    per-track by position — see _attach_deltas.

    Raises ValueError for the same files the engine rejects (type 2, 6/8 mixed
    with other meters) so the user hears about it at LOAD, not at HUMANISE.
    """
    mid = mido.MidiFile(str(path))
    if mid.type == 2:
        raise ValueError("Type 2 MIDI files are not supported")
    meter = detect_meter(mid)  # raises for 6/8 mixed with other signatures

    tempo_map = build_tempo_map(mid)
    ppq = mid.ticks_per_beat

    hits: list[dict] = []
    time_sigs: list[tuple[int, int, int]] = []
    other_channel_drum_hits = 0
    end_tick = 0

    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            end_tick = max(end_tick, abs_tick)
            if msg.type == "time_signature":
                time_sigs.append((abs_tick, msg.numerator, msg.denominator))
            elif (
                msg.type == "note_on"
                and msg.velocity > 0
                and hasattr(msg, "note")
                and msg.note in TD11_TO_GROUP
            ):
                if all_channels or msg.channel == DRUM_CHANNEL:
                    hits.append({
                        "track": ti,
                        "tick": abs_tick,
                        "ms": ticks_to_ms_with_map(0, abs_tick, tempo_map, ppq),
                        "lane": TD11_TO_GROUP[msg.note],
                        "note": msg.note,
                        "velocity": msg.velocity,
                    })
                else:
                    other_channel_drum_hits += 1

    bars, beats = _bars_and_beats(time_sigs, end_tick, ppq, tempo_map)

    return {
        "hits": hits,
        "lanes": LANE_ORDER,
        "bars": bars,
        "beats": beats,
        "duration_ms": ticks_to_ms_with_map(0, end_tick, tempo_map, ppq),
        "meter": meter,
        "ppq": ppq,
        "bpm": round(60_000_000 / tempo_map[0][1], 2),
        "num_tracks": len(mid.tracks),
        "other_channel_drum_hits": other_channel_drum_hits,
    }


def _attach_deltas(original: dict, humanised: dict) -> None:
    """Annotate humanised hits with their per-hit shift vs the original.

    The engine never drops, adds, or reorders note_ons within a track, so the
    k-th drum hit of a track in the output corresponds to the k-th in the
    input. Pairs per track by position; on any count mismatch (shouldn't
    happen) the annotation is skipped rather than guessed.
    """
    by_track_orig: dict[int, list[dict]] = defaultdict(list)
    by_track_new: dict[int, list[dict]] = defaultdict(list)
    for h in original["hits"]:
        by_track_orig[h["track"]].append(h)
    for h in humanised["hits"]:
        by_track_new[h["track"]].append(h)

    if {t: len(v) for t, v in by_track_orig.items()} != {t: len(v) for t, v in by_track_new.items()}:
        return
    for track, new_hits in by_track_new.items():
        for orig_hit, new_hit in zip(by_track_orig[track], new_hits):
            new_hit["orig_ms"] = orig_hit["ms"]
            new_hit["delta_ms"] = new_hit["ms"] - orig_hit["ms"]
            new_hit["delta_vel"] = new_hit["velocity"] - orig_hit["velocity"]


def _bundled_profile_path():
    return files("wobblemidi.profiles").joinpath("rock.json")


class Session:
    """One GUI session: the loaded profile, the original file, and renders.

    Renders always re-run the engine from the ORIGINAL loaded file — never from
    a previous render (re-humanising output destroys the groove; roadmap
    "preserve-intent"). Engine calls serialise through one lock because
    humanise() seeds the process-global numpy RNG.
    """

    def __init__(self, profile_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="wobblemidi-gui-"))
        self._render_count = 0
        self.original_path: Path | None = None
        self.render_path: Path | None = None
        self.seed: int | None = None
        self.render_params: dict | None = None
        # Single-level undo: the previous render's (path, seed, params). undo()
        # swaps it with the current one — pressing twice is redo. Deliberately
        # no history stack (fast workflow, no state pile-up).
        self._prev_render: tuple[Path, int, dict] | None = None
        if profile_path is not None:
            self.profiles = load_profile(profile_path)
        else:
            with as_file(_bundled_profile_path()) as p:
                self.profiles = load_profile(p)

    # -- lifecycle ---------------------------------------------------------

    def cleanup(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # -- API ----------------------------------------------------------------

    def load(self, path: str | Path) -> dict:
        try:
            display = parse_for_display(path, all_channels=False)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # corrupt/unreadable file — mido raises variously
            return {"ok": False, "error": f"Could not read MIDI file: {exc}"}

        self.original_path = Path(path)
        self.render_path = None
        self.seed = None
        self.render_params = None
        self._prev_render = None

        warnings = []
        if not display["hits"]:
            if display["other_channel_drum_hits"]:
                warnings.append("no_drum_hits_channel10")
            else:
                warnings.append("no_drum_hits")
        return {
            "ok": True,
            "file_name": Path(path).name,
            "original": display,
            "warnings": warnings,
        }

    @staticmethod
    def _clean_params(params: dict) -> dict:
        lane_intensity = {
            str(k): float(v)
            for k, v in (params.get("lane_intensity") or {}).items()
        }
        return {
            "intensity": float(params.get("intensity", 0.35)),
            "tightness": float(params.get("tightness", 0.4)),
            "lean": float(params.get("lean", 0.0)),
            "all_channels": bool(params.get("all_channels", False)),
            "lane_intensity": lane_intensity,
        }

    def _render_response(self, params: dict) -> dict:
        original = parse_for_display(self.original_path,
                                     all_channels=params["all_channels"])
        humanised = parse_for_display(self.render_path,
                                      all_channels=params["all_channels"])
        _attach_deltas(original, humanised)
        return {
            "ok": True,
            "original": original,
            "humanised": humanised,
            "seed": self.seed,
            "params": params,
            "can_undo": self._prev_render is not None,
        }

    def humanise_current(self, params: dict) -> dict:
        """Render with the given params. Every render is a fresh take (new
        random seed) — there is deliberately no same-seed reuse."""
        if self.original_path is None:
            return {"ok": False, "error": "No MIDI file loaded."}

        p = self._clean_params(params)
        seed = random.randrange(2**32)

        with self._lock:
            self._render_count += 1
            out = self._tmpdir / f"render_{self._render_count:03d}.mid"
            try:
                humanise(
                    input_path=self.original_path,
                    output_path=out,
                    profiles=self.profiles,
                    intensity=p["intensity"],
                    seed=seed,
                    phi=p["tightness"],
                    all_channels=p["all_channels"],
                    push_amount=p["lean"],
                    intensity_by_group=p["lane_intensity"] or None,
                )
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            except Exception as exc:
                return {"ok": False, "error": f"Humanise failed: {exc}"}

        if self.render_path is not None:
            self._prev_render = (self.render_path, self.seed, self.render_params)
        self.render_path = out
        self.seed = seed
        self.render_params = p
        return self._render_response(p)

    def undo(self) -> dict:
        """Swap the current render with the previous one (single level — undo
        twice to redo). Returns the same shape as humanise_current, with the
        restored params so the UI can snap its controls back."""
        if self._prev_render is None:
            return {"ok": False, "error": "Nothing to undo yet."}
        prev_path, prev_seed, prev_params = self._prev_render
        self._prev_render = (self.render_path, self.seed, self.render_params)
        self.render_path, self.seed, self.render_params = (
            prev_path, prev_seed, prev_params)
        return self._render_response(self.render_params)

    def export_to(self, dest: str | Path) -> dict:
        if self.render_path is None:
            return {"ok": False, "error": "Nothing to export — humanise first."}
        try:
            shutil.copyfile(self.render_path, dest)
        except OSError as exc:
            return {"ok": False, "error": f"Export failed: {exc}"}
        return {"ok": True, "path": str(dest)}
