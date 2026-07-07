"""Tier 2 port-vs-reference equivalence runner (porting contract, Tier 2(b)).

Grades a port's outputs on the golden input fixtures against the reference engine
using the validation harness's metric suite — distributional equivalence, seeds free
to differ. This is the port gate: green here (plus the Tier 2(a) harness envelope)
means the port is correct. See wobblemidi_porting_contract.md.

How it works
------------
Each golden vector (minus CLI/seed duplicates) plus the runner-owned full-kit
coverage fixture (tests/tier2/t2_full_kit.mid — covers all 22 TD-11 notes, which
the golden inputs do not) is a *cell*: one (input, pinned params) point. The
fixtures are short, so equivalence is only measurable on pools: the port must
provide K_RUNS=32 outputs per cell, each an independent-seed run; the runner
generates its own 32-run reference pool (pinned seeds REFERENCE_SEED_BASE+i) and
compares pooled per-instrument distributions and per-run statistics:

  timing offsets from input position (W1/KS/mean/sigma), velocity deltas from
  input (W1/mean/sigma), drum note_off placement (W1), per-run kit-wide timing
  lag-1 autocorr, per-instrument velocity lag-1, within-(position, role) velocity
  sigma (roles fixed from the INPUT velocities), zero micro-jump mass, same-slot
  cross-instrument gap sigma, and sub-12ms cluster-pair gap deviation.

Metric functions are imported from scripts/validate.py — the same scoring core the
harness uses. Thresholds are NOT chosen here: they are calibrated empirically
(scripts/calibrate_tier2.py — null variance of reference-vs-itself plus a mutation
battery) and locked in calibration/tier2_thresholds.json. Verdict: every gated
comparison within its band, plus per-metric-family aggregate z-scores. Exit 0/1.

Candidate directory layout
--------------------------
    CANDIDATE_DIR/<cell_id>/*.mid     # exactly K_RUNS files per distributional cell
                                      # >=1 file per identity cell (f9_empty_default,
                                      # f1_intensity00 — checked structurally)

Usage
-----
    python scripts/compare_port.py verify CANDIDATE_DIR        # the gate
    python scripts/compare_port.py self-null                   # reference vs itself
    python scripts/compare_port.py fixture [--check]           # t2_full_kit builder
    python scripts/compare_port.py dump-reference OUT_DIR      # pools for port devs
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
import tempfile
import warnings
from pathlib import Path

import click
import mido
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from wobblemidi import humanise as humanise_mod
from wobblemidi.humanise import humanise
from wobblemidi.midi_utils import (
    TD11_TO_GROUP,
    build_tempo_map,
    detect_meter,
    grid_position_in_bar,
    is_four_four,
    quantise_to_grid,
    ticks_to_ms_with_map,
)

from validate import (
    GROUP_ORDER,
    _jumps,
    _lag1,
    _wrole_sigma,
    _xgap_sigma,
    role_labels,
    signed_offset_ms,
)
from verify_golden import load_bundled_profile, load_manifest

GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
TIER2_DIR = REPO_ROOT / "tests" / "tier2"
FULL_KIT_FIXTURE = TIER2_DIR / "t2_full_kit.mid"
THRESHOLDS_DEFAULT = REPO_ROOT / "calibration" / "tier2_thresholds.json"

# Runs per engine per cell. Locked together with the thresholds: null variance
# (and therefore every band) is a function of pool size, so a candidate pool of a
# different size is not comparable against the calibrated bands.
K_RUNS = 32
# The runner's reference pool seeds are PINNED (BASE+0 .. BASE+K-1): thresholds are
# calibrated as "this fixed reference pool vs an independent pool", which is exactly
# the comparison the runner performs. Port seeds are free.
REFERENCE_SEED_BASE = 777_000
MIN_POOLED_HITS = 100   # per-group rows need n_hits_per_run * K_RUNS >= this

# Snapshot at import time so a calibration mutant that patches the engine constant
# during run GENERATION can never change how the scorer identifies input clusters.
WINDOW_MS = float(humanise_mod.COUPLE_WINDOW_MS)

# zjump mass is lattice-valued (multiples of 1 / n_jumps); below this many hits
# per run the per-group null quantiles are too coarse to gate on (calibration
# round 1 finding: t2 tom_low at 16 hits/run drove held-out false failures).
ZJUMP_MIN_HITS = 32
# Per-note distribution gates run on the full-kit cell only: mapping errors are
# per-note phenomena and group-level pooling dilutes low-share variants
# (calibration round 1: note 22 → hihat_open was detected at ratio 1.05 only).
PER_NOTE_MIN_HITS = 4

# Golden vectors that are not Tier 2 cells.
SKIPPED_VECTORS = {
    "f1_cli": "CLI plumbing lock — no port analogue; params identical to f1_default",
    "f1_seed7": "seed-sensitivity lock — duplicate of f1_default once seeds are free",
}
# Identity contracts: the reference output is byte-identical to the input, so the
# port is checked for structural equality, not distributions.
IDENTITY_VECTORS = {"f9_empty_default", "f1_intensity00"}

FULL_KIT_CELL_ID = "t2_full_kit"

POOLED_TIMING_METRICS = ["off_w1", "off_ks", "off_mean_d", "off_sigma_d"]
POOLED_VEL_METRICS = ["veld_w1", "veld_mean_d", "veld_sigma_d"]
RUNSTAT_TIMING_METRICS = ["t_lag1_d", "xgap_sigma_d"]      # ALL row only
RUNSTAT_VEL_METRICS = ["v_lag1_d", "wrole_sigma_d", "zjump_d"]


# ---------------------------------------------------------------------------
# Full-kit coverage fixture (runner-owned; NOT part of the golden byte contract)
# ---------------------------------------------------------------------------

_FK_PPQ = 480
_FK_TEMPO = 500_000     # 120 BPM


def _full_kit_events() -> list[tuple[int, int, int, int]]:
    """(tick, note, velocity, duration) covering all 22 TD-11 notes.

    16 bars, 4/4, 120 BPM: closed-hat section (bars 0-7, notes 42/44/22 with the
    46/26 open voices), ride section (bars 8-15, notes 51/59/53), kick 36 with a
    3-level velocity pattern, snare 38/40 backbeats + 37 side-stick ghosts
    (two-cluster velocities engage relative tiering), rotating crash 49/55/57/52
    on kick downbeats (same-tick chords), and tom fills 48/50/45/47/43/58 on
    bars 3/7/11/15.
    """
    S, E, BAR = _FK_PPQ // 4, _FK_PPQ // 2, 4 * _FK_PPQ
    ev: list[tuple[int, int, int, int]] = []
    closed, rides, crashes = [42, 44, 22], [51, 59, 53], [49, 55, 57, 52]
    fill_bars = {3, 7, 11, 15}
    for bar in range(16):
        b = bar * BAR
        fill = bar in fill_bars
        timekeeper = closed if bar < 8 else rides
        for i in range(8):                          # 8th-note timekeeping slots
            if fill and i >= 4:
                continue                            # the fill owns beats 3-4
            t = b + i * E
            if i == 3:                              # "2&": open-hat voice
                ev.append((t, 46 if bar % 2 == 0 else 26, 90, 100))
            else:
                ev.append((t, timekeeper[(bar * 8 + i) % 3],
                           84 if i % 2 == 0 else 62, 40))
        ev.append((b, 36, 112, 60))                             # kick beat 1
        if not fill:
            ev.append((b + 2 * _FK_PPQ, 36, 104, 60))           # kick beat 3
            if bar % 2 == 0:
                ev.append((b + 2 * _FK_PPQ + E, 36, 96, 60))    # kick "3&"
        ev.append((b + _FK_PPQ, 38, 108, 60))                   # snare beat 2
        if not fill:
            ev.append((b + 3 * _FK_PPQ, 40, 102, 60))           # snare beat 4
        ev.append((b + 3 * S, 37, 34, 30))                      # side-stick ghost
        if fill:
            notes16 = ([48, 48, 50, 50, 45, 45, 47, 47] if bar in (3, 11)
                       else [43, 43, 58, 58, 43, 58, 43, 58])
            for k, n in enumerate(notes16):
                ev.append((b + 2 * _FK_PPQ + k * S, n, 70 + (k % 2) * 28, 55))
        ev.append((b, crashes[bar % 4], 110, 240))              # crash+kick+hat chord
    return ev


def build_full_kit_midi() -> mido.MidiFile:
    events = _full_kit_events()

    note_counts: dict[int, int] = {}
    group_counts: dict[str, int] = {}
    for _t, note, _v, _d in events:
        note_counts[note] = note_counts.get(note, 0) + 1
        g = TD11_TO_GROUP[note]
        group_counts[g] = group_counts.get(g, 0) + 1
    missing = set(TD11_TO_GROUP) - set(note_counts)
    assert not missing, f"full-kit fixture missing notes: {sorted(missing)}"
    thin_notes = {n: c for n, c in note_counts.items() if c < 4}
    assert not thin_notes, f"full-kit notes with < 4 hits: {thin_notes}"
    thin_groups = {g: c for g, c in group_counts.items() if c < 8}
    assert not thin_groups, f"full-kit groups with < 8 hits: {thin_groups}"

    seq: list[tuple[int, int, mido.Message | mido.MetaMessage]] = [
        (0, 0, mido.MetaMessage("set_tempo", tempo=_FK_TEMPO, time=0)),
    ]
    for t, note, vel, dur in events:
        seq.append((t, 2, mido.Message("note_on", channel=9, note=note,
                                       velocity=vel, time=0)))
        seq.append((t + dur, 1, mido.Message("note_off", channel=9, note=note,
                                             velocity=0, time=0)))
    seq.sort(key=lambda e: (e[0], e[1]))    # same-tick order: metas, offs, ons

    track = mido.MidiTrack()
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4,
                                  clocks_per_click=24,
                                  notated_32nd_notes_per_beat=8, time=0))
    prev = 0
    for t, _, msg in seq:
        track.append(msg.copy(time=t - prev))
        prev = t
    track.append(mido.MetaMessage("end_of_track", time=0))

    mid = mido.MidiFile(type=0, ticks_per_beat=_FK_PPQ)
    mid.tracks.append(track)
    return mid


def full_kit_bytes() -> bytes:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fk.mid"
        build_full_kit_midi().save(str(p))
        return p.read_bytes()


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Cell:
    id: str
    input_path: Path
    params: dict            # humanise() kwargs, seed removed
    kind: str               # "dist" | "identity"


def load_cells(manifest: dict | None = None) -> list[Cell]:
    """Tier 2 cell list: manifest vectors (deduplicated) + the full-kit cell."""
    if manifest is None:
        manifest = load_manifest()
    cells: list[Cell] = []
    default_params: dict | None = None
    for entry in manifest["vectors"]:
        vid = entry["id"]
        if vid in SKIPPED_VECTORS or "cli_args" in entry:
            continue
        params = dict(entry["params"])
        params.pop("seed", None)
        if vid == "f1_default":
            default_params = dict(params)
        cells.append(Cell(
            id=vid,
            input_path=GOLDEN_DIR / entry["input"],
            params=params,
            kind="identity" if vid in IDENTITY_VECTORS else "dist",
        ))
    assert default_params is not None, "manifest has no f1_default vector"
    cells.append(Cell(
        id=FULL_KIT_CELL_ID,
        input_path=FULL_KIT_FIXTURE,
        params=default_params,
        kind="dist",
    ))
    return cells


# ---------------------------------------------------------------------------
# Input preparation and output extraction
# ---------------------------------------------------------------------------

def _is_hit(msg, all_channels: bool) -> bool:
    return (
        msg.type == "note_on"
        and msg.velocity > 0
        and hasattr(msg, "note")
        and (all_channels or msg.channel == humanise_mod.DRUM_CHANNEL)
        and msg.note in TD11_TO_GROUP
    )


def _is_drum_off(msg, all_channels: bool) -> bool:
    if not (hasattr(msg, "note") and msg.note in TD11_TO_GROUP):
        return False
    if not (all_channels or msg.channel == humanise_mod.DRUM_CHANNEL):
        return False
    return msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)


def _walk(mid: mido.MidiFile):
    """Yield (track_index, abs_tick, msg) in file order."""
    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            yield ti, abs_tick, msg


@dataclasses.dataclass
class CellData:
    """Static, input-derived facts for one cell (shared by every run/engine)."""
    cell: Cell
    ppq: int
    tempo_map: list[tuple[int, int]]
    n_tracks: int
    # hit arrays, in track-walk order (the order humanise preserves per track)
    track: np.ndarray
    channel: np.ndarray
    notes: np.ndarray
    in_abs: np.ndarray
    in_vel: np.ndarray
    groups: np.ndarray
    slot: np.ndarray        # quantised grid tick
    pos: np.ndarray         # 16th position in a 4/4 bar; 0 for non-4/4 files
    roles: np.ndarray       # fixed per-hit labels from the INPUT velocities
    ord: np.ndarray         # merged time-major order index
    in_t: np.ndarray        # input ms positions
    # drum note_offs (track-walk order), for off-placement equivalence
    off_track: np.ndarray
    off_notes: np.ndarray
    off_channel: np.ndarray
    off_abs: np.ndarray
    # consecutive time-major hit pairs with 0 < gap <= WINDOW_MS (input clusters)
    window_pairs: list[tuple[int, int]]
    # structural expectations
    melodic_notes: list     # per-track normalised non-drum-channel note events
    metas: list             # (track, abs, type, key params) for set_tempo/time_signature
    gated_groups: list[str]
    group_counts: dict      # hits per run per group
    note_masks: dict        # full-kit cell only: "n<note>" -> hit mask (per-note gates)
    timing_scored: bool
    velocity_scored: bool


def _norm_note_event(abs_tick: int, msg) -> tuple | None:
    """Normalised note event: encoding-tolerant (note_on vel 0 == note_off)."""
    if msg.type == "note_on" and msg.velocity > 0:
        return (abs_tick, "on", msg.channel, msg.note, msg.velocity)
    if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
        return (abs_tick, "off", msg.channel, msg.note)
    return None


def _extract_metas(mid: mido.MidiFile) -> list[tuple]:
    out = []
    for ti, abs_tick, msg in _walk(mid):
        if msg.type == "set_tempo":
            out.append((ti, abs_tick, "set_tempo", msg.tempo))
        elif msg.type == "time_signature":
            out.append((ti, abs_tick, "time_signature", msg.numerator, msg.denominator))
    return out


def _extract_melodic(mid: mido.MidiFile) -> list[list[tuple]]:
    """Per-track normalised note events NOT on the drum channel."""
    out: list[list[tuple]] = [[] for _ in mid.tracks]
    for ti, abs_tick, msg in _walk(mid):
        if hasattr(msg, "note") and msg.channel != humanise_mod.DRUM_CHANNEL:
            ev = _norm_note_event(abs_tick, msg)
            if ev is not None:
                out[ti].append(ev)
    return out


def prepare_cell(cell: Cell) -> CellData:
    mid = mido.MidiFile(str(cell.input_path))
    ppq = mid.ticks_per_beat
    tempo_map = build_tempo_map(mid)
    all_channels = bool(cell.params.get("all_channels", False))
    meter = detect_meter(mid)
    grid = "8" if meter == "6/8" else "16"
    use_grid_pos = meter != "6/8" and is_four_four(mid)

    tr, ch, notes, in_abs, in_vel = [], [], [], [], []
    off_tr, off_notes, off_ch, off_abs = [], [], [], []
    for ti, abs_tick, msg in _walk(mid):
        if _is_hit(msg, all_channels):
            tr.append(ti)
            ch.append(msg.channel)
            notes.append(msg.note)
            in_abs.append(abs_tick)
            in_vel.append(msg.velocity)
        elif _is_drum_off(msg, all_channels):
            off_tr.append(ti)
            off_notes.append(msg.note)
            off_ch.append(msg.channel)
            off_abs.append(abs_tick)

    tr = np.array(tr, dtype=int)
    ch = np.array(ch, dtype=int)
    notes = np.array(notes, dtype=int)
    in_abs = np.array(in_abs, dtype=int)
    in_vel = np.array(in_vel, dtype=int)
    n = len(notes)
    groups = np.array([TD11_TO_GROUP[x] for x in notes]) if n else np.array([], dtype=object)
    slot = np.array([quantise_to_grid(int(a), ppq, grid) for a in in_abs], dtype=int)
    pos = (np.array([grid_position_in_bar(int(s), ppq) for s in slot], dtype=int)
           if use_grid_pos else np.zeros(n, dtype=int))

    roles = np.empty(n, dtype=object)
    for g in np.unique(groups):
        m = groups == g
        roles[m] = role_labels(in_vel[m])

    # merged time-major order (abs, track, walk-index) — the engine's stream order
    order = np.lexsort((np.arange(n), tr, in_abs))
    ord_idx = np.empty(n, dtype=int)
    ord_idx[order] = np.arange(n)

    in_t = np.array([ticks_to_ms_with_map(0, int(a), tempo_map, ppq) for a in in_abs])

    window_pairs: list[tuple[int, int]] = []
    for a, b in zip(order[:-1], order[1:]):
        gap = in_t[b] - in_t[a]
        if 0.0 < gap <= WINDOW_MS:
            window_pairs.append((int(a), int(b)))

    counts = {g: int((groups == g).sum()) for g in np.unique(groups)}
    gated = [g for g in GROUP_ORDER
             if counts.get(g, 0) * K_RUNS >= MIN_POOLED_HITS]

    note_masks: dict[str, np.ndarray] = {}
    if cell.id == FULL_KIT_CELL_ID:
        for note in sorted(np.unique(notes)):
            mask = notes == note
            if int(mask.sum()) >= PER_NOTE_MIN_HITS:
                note_masks[f"n{int(note)}"] = mask

    return CellData(
        cell=cell, ppq=ppq, tempo_map=tempo_map, n_tracks=len(mid.tracks),
        track=tr, channel=ch, notes=notes, in_abs=in_abs, in_vel=in_vel,
        groups=groups, slot=slot, pos=pos, roles=roles, ord=ord_idx, in_t=in_t,
        off_track=np.array(off_tr, dtype=int), off_notes=np.array(off_notes, dtype=int),
        off_channel=np.array(off_ch, dtype=int), off_abs=np.array(off_abs, dtype=int),
        window_pairs=window_pairs,
        melodic_notes=[] if all_channels else _extract_melodic(mid),
        metas=_extract_metas(mid),
        gated_groups=gated,
        group_counts=counts,
        note_masks=note_masks,
        timing_scored=not bool(cell.params.get("velocity_only", False)),
        velocity_scored=not bool(cell.params.get("timing_only", False)),
    )


def read_output(path: Path, cd: CellData) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract one run's (hit_abs, hit_vel, drum_off_abs) aligned to cd's order,
    plus structural violations. Alignment failure is itself a violation and the
    arrays come back empty."""
    violations: list[str] = []
    try:
        mid = mido.MidiFile(str(path))
    except Exception as e:  # noqa: BLE001 — candidate files are external input
        return np.array([]), np.array([]), np.array([]), [f"unreadable MIDI: {e}"]

    all_channels = bool(cd.cell.params.get("all_channels", False))
    if len(mid.tracks) != cd.n_tracks:
        violations.append(
            f"track count {len(mid.tracks)} != input's {cd.n_tracks} "
            "(Tier 2 requires the port to preserve track structure)")
        return np.array([]), np.array([]), np.array([]), violations

    tr, ch, notes, abss, vels = [], [], [], [], []
    off_tr, off_notes, off_ch, off_abs = [], [], [], []
    for ti, abs_tick, msg in _walk(mid):
        if _is_hit(msg, all_channels):
            tr.append(ti)
            ch.append(msg.channel)
            notes.append(msg.note)
            abss.append(abs_tick)
            vels.append(msg.velocity)
        elif _is_drum_off(msg, all_channels):
            off_tr.append(ti)
            off_notes.append(msg.note)
            off_ch.append(msg.channel)
            off_abs.append(abs_tick)

    if (len(notes) != len(cd.notes)
            or not np.array_equal(np.array(notes, dtype=int), cd.notes)
            or not np.array_equal(np.array(tr, dtype=int), cd.track)
            or not np.array_equal(np.array(ch, dtype=int), cd.channel)):
        violations.append(
            f"hit misalignment: {len(notes)} hits vs input's {len(cd.notes)} "
            "(or note/track/channel sequence differs) — humanisation must "
            "preserve note identity, count and per-track order")
        return np.array([]), np.array([]), np.array([]), violations

    if (len(off_notes) != len(cd.off_notes)
            or not np.array_equal(np.array(off_notes, dtype=int), cd.off_notes)
            or not np.array_equal(np.array(off_tr, dtype=int), cd.off_track)
            or not np.array_equal(np.array(off_ch, dtype=int), cd.off_channel)):
        violations.append("drum note_off misalignment vs input")
        off_abs = list(cd.off_abs)   # keep scoring alive; the violation still fails

    out_abs = np.array(abss, dtype=int)
    out_vel = np.array(vels, dtype=int)

    if not cd.timing_scored and not np.array_equal(out_abs, cd.in_abs):
        violations.append("velocity_only cell: note positions differ from input")
    if not cd.velocity_scored and not np.array_equal(out_vel, cd.in_vel):
        violations.append("timing_only cell: velocities differ from input")
    if len(out_vel) and (out_vel.min() < 1 or out_vel.max() > 127):
        violations.append("velocities outside [1, 127]")
    if len(out_abs) and out_abs.min() < 0:
        violations.append("negative output tick")

    if not all_channels:
        got_melodic = _extract_melodic(mid)
        if got_melodic != cd.melodic_notes:
            violations.append("non-drum-channel note events differ from input "
                              "(must pass through untouched without --all-channels)")
    if _extract_metas(mid) != cd.metas:
        violations.append("tempo / time-signature events differ from input")

    return out_abs, out_vel, np.array(off_abs, dtype=int), violations


# ---------------------------------------------------------------------------
# Per-run results and pool scoring
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RunResult:
    off: np.ndarray         # ms deviation from input position, per hit
    veld: np.ndarray        # velocity delta from input, per hit
    vel: np.ndarray         # output velocity
    offd: np.ndarray        # drum note_off ms deviation from input
    cdev: np.ndarray        # window-pair gap deviations (ms)
    runstats: dict[tuple[str, str], float]
    violations: list[str]


def _run_frame(cd: CellData, off, vel, t) -> pd.DataFrame:
    return pd.DataFrame({
        "take": 0,
        "group": cd.groups,
        "pos": cd.pos,
        "slot": cd.slot,
        "role": cd.roles,
        "ord": cd.ord,
        "off": off,
        "vel": vel,
        "t": t,
    })


def make_run_result(cd: CellData, out_abs: np.ndarray, out_vel: np.ndarray,
                    out_off_abs: np.ndarray, violations: list[str]) -> RunResult:
    n = len(cd.notes)
    if len(out_abs) != n:       # misaligned run — carry the violation only
        empty = np.array([])
        return RunResult(empty, empty, empty, empty, empty, {}, violations)

    off = np.array([
        signed_offset_ms(int(a), int(b), cd.tempo_map, cd.ppq)
        for a, b in zip(cd.in_abs, out_abs)
    ]) if n else np.array([])
    veld = (out_vel - cd.in_vel).astype(float)
    t = np.array([ticks_to_ms_with_map(0, int(a), cd.tempo_map, cd.ppq)
                  for a in out_abs]) if n else np.array([])
    offd = np.array([
        signed_offset_ms(int(a), int(b), cd.tempo_map, cd.ppq)
        for a, b in zip(cd.off_abs, out_off_abs)
    ]) if len(cd.off_abs) and len(out_off_abs) == len(cd.off_abs) else np.array([])
    cdev = np.array([
        (t[j] - t[i]) - (cd.in_t[j] - cd.in_t[i]) for i, j in cd.window_pairs
    ]) if cd.window_pairs and n else np.array([])

    runstats: dict[tuple[str, str], float] = {}
    if n:
        df = _run_frame(cd, off, out_vel.astype(float), t)
        for g in ["ALL"] + cd.gated_groups:
            sub = df if g == "ALL" else df[df["group"] == g]
            if cd.velocity_scored:
                jumps = _jumps(sub, "vel")
                runstats[(g, "zjump")] = (
                    float((jumps <= 1.0).mean()) if len(jumps) else float("nan"))
                runstats[(g, "wrole_sigma")] = _wrole_sigma(sub)
                runstats[(g, "v_lag1")] = _lag1(
                    sub, "vel", kitwide=False, demean_keys=("take", "group", "pos"))
            if cd.timing_scored and g == "ALL":
                runstats[(g, "t_lag1")] = _lag1(sub, "off", kitwide=True)
                runstats[(g, "xgap_sigma")] = _xgap_sigma(sub)
    return RunResult(off=off, veld=veld, vel=out_vel.astype(float),
                     offd=offd, cdev=cdev, runstats=runstats, violations=violations)


def generate_runs(cd: CellData, profile, seeds: list[int],
                  params: dict | None = None, strict: bool = True) -> list[RunResult]:
    """Run the (in-process) engine once per seed and score each run.

    strict=True (the runner's reference pool): any structural violation is a bug
    in this script or the engine — raise. strict=False (calibration mutants):
    keep the violation on the RunResult so the verdict records it."""
    runs = []
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.mid"
        for seed in seeds:
            humanise(cd.cell.input_path, out, profile, seed=seed,
                     **(params if params is not None else cd.cell.params))
            out_abs, out_vel, out_off, viol = read_output(out, cd)
            if viol and strict:
                raise RuntimeError(
                    f"reference engine violated its own structural contract on "
                    f"{cd.cell.id} (seed {seed}): {viol}")
            runs.append(make_run_result(cd, out_abs, out_vel, out_off, viol))
    return runs


def load_candidate_runs(cd: CellData, files: list[Path]) -> list[RunResult]:
    runs = []
    for p in files:
        out_abs, out_vel, out_off, viol = read_output(p, cd)
        viol = [f"{p.name}: {v}" for v in viol]
        runs.append(make_run_result(cd, out_abs, out_vel, out_off, viol))
    return runs


def _mean_runstat(runs: list[RunResult], key: tuple[str, str]) -> float:
    vals = np.array([r.runstats.get(key, float("nan")) for r in runs], dtype=float)
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def _pool(runs: list[RunResult], attr: str, mask: np.ndarray | None = None) -> np.ndarray:
    arrs = []
    for r in runs:
        a = getattr(r, attr)
        if len(a) == 0:
            continue
        arrs.append(a if mask is None else a[mask])
    return np.concatenate(arrs) if arrs else np.array([])


def _dist_metrics(a: np.ndarray, b: np.ndarray, prefix: str) -> dict[str, float]:
    if len(a) == 0 or len(b) == 0:
        return {f"{prefix}_w1": float("nan"), f"{prefix}_ks": float("nan"),
                f"{prefix}_mean_d": float("nan"), f"{prefix}_sigma_d": float("nan")}
    return {
        f"{prefix}_w1": float(wasserstein_distance(a, b)),
        f"{prefix}_ks": float(ks_2samp(a, b).statistic),
        f"{prefix}_mean_d": float(abs(a.mean() - b.mean())),
        f"{prefix}_sigma_d": float(abs(a.std(ddof=1) - b.std(ddof=1)))
        if len(a) > 1 and len(b) > 1 else float("nan"),
    }


def score_pools(cd: CellData, runs_a: list[RunResult],
                runs_b: list[RunResult]) -> dict[str, float]:
    """All equivalence metrics for candidate pool A vs reference pool B.

    Keys are "<group>|<metric>". Symmetric in A/B except for nothing — order is
    conventional (candidate first)."""
    out: dict[str, float] = {}
    for g in ["ALL"] + cd.gated_groups:
        mask = None if g == "ALL" else (cd.groups == g)
        if cd.timing_scored:
            d = _dist_metrics(_pool(runs_a, "off", mask), _pool(runs_b, "off", mask), "off")
            if g != "ALL":
                del d["off_ks"]     # keep KS at ALL level only (per-group adds little)
            out.update({f"{g}|{k}": v for k, v in d.items()})
        if cd.velocity_scored:
            d = _dist_metrics(_pool(runs_a, "veld", mask), _pool(runs_b, "veld", mask),
                              "veld")
            del d["veld_ks"]
            out.update({f"{g}|{k}": v for k, v in d.items()})
            stats = [("v_lag1", "v_lag1_d"), ("wrole_sigma", "wrole_sigma_d")]
            if g == "ALL" or cd.group_counts.get(g, 0) >= ZJUMP_MIN_HITS:
                stats.append(("zjump", "zjump_d"))
            for stat, name in stats:
                a = _mean_runstat(runs_a, (g, stat))
                b = _mean_runstat(runs_b, (g, stat))
                out[f"{g}|{name}"] = float(abs(a - b))
        if cd.timing_scored and g == "ALL":
            for stat, name in [("t_lag1", "t_lag1_d"), ("xgap_sigma", "xgap_sigma_d")]:
                a = _mean_runstat(runs_a, (g, stat))
                b = _mean_runstat(runs_b, (g, stat))
                out[f"{g}|{name}"] = float(abs(a - b))
    if cd.timing_scored and len(cd.off_abs):
        a, b = _pool(runs_a, "offd"), _pool(runs_b, "offd")
        out["ALL|offd_w1"] = (float(wasserstein_distance(a, b))
                              if len(a) and len(b) else float("nan"))
    if cd.timing_scored and cd.window_pairs:
        a, b = _pool(runs_a, "cdev"), _pool(runs_b, "cdev")
        out["ALL|cdev_w1"] = (float(wasserstein_distance(a, b))
                              if len(a) and len(b) else float("nan"))
    # Per-note pooled distributions (full-kit cell only): mapping errors are
    # per-note, and group pooling dilutes low-share variants like 22/26/44.
    for note_key, mask in cd.note_masks.items():
        if cd.timing_scored:
            a, b = _pool(runs_a, "off", mask), _pool(runs_b, "off", mask)
            out[f"{note_key}|off_w1"] = (float(wasserstein_distance(a, b))
                                         if len(a) and len(b) else float("nan"))
        if cd.velocity_scored:
            a, b = _pool(runs_a, "veld", mask), _pool(runs_b, "veld", mask)
            out[f"{note_key}|veld_w1"] = (float(wasserstein_distance(a, b))
                                          if len(a) and len(b) else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Identity cells (structural equality)
# ---------------------------------------------------------------------------

def check_identity(input_path: Path, output_path: Path) -> tuple[list[str], list[str]]:
    """(violations, warnings): note events + tempo/time-sig must match the input
    exactly; other meta differences are warnings (a port's writer may re-encode)."""
    violations: list[str] = []
    warnings_: list[str] = []
    try:
        got = mido.MidiFile(str(output_path))
    except Exception as e:  # noqa: BLE001
        return [f"unreadable MIDI: {e}"], []
    exp = mido.MidiFile(str(input_path))
    if len(got.tracks) != len(exp.tracks):
        return [f"track count {len(got.tracks)} != input's {len(exp.tracks)}"], []

    def _notes(mid):
        out = [[] for _ in mid.tracks]
        for ti, abs_tick, msg in _walk(mid):
            ev = _norm_note_event(abs_tick, msg)
            if ev is not None:
                out[ti].append(ev)
        return out

    if _notes(got) != _notes(exp):
        violations.append("note events differ from input (identity cell: the "
                          "engine must not change anything here)")
    if _extract_metas(got) != _extract_metas(exp):
        violations.append("tempo / time-signature events differ from input")

    def _other_metas(mid):
        out = []
        for ti, abs_tick, msg in _walk(mid):
            if msg.is_meta and msg.type not in ("set_tempo", "time_signature"):
                out.append((ti, abs_tick, msg.type))
        return out

    if _other_metas(got) != _other_metas(exp):
        warnings_.append("other meta events differ from input (warning only)")
    return violations, warnings_


# ---------------------------------------------------------------------------
# Thresholds and verdict
# ---------------------------------------------------------------------------

def load_thresholds(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


@dataclasses.dataclass
class Verdict:
    passed: bool
    failures: list[str]
    warnings: list[str]
    n_gated: int
    n_checked: int
    aggregates: dict[str, dict[str, float]]
    rows: list[tuple[str, float, float, float]]   # (key, value, threshold, ratio)


def evaluate(scores_by_cell: dict[str, dict[str, float]],
             violations_by_cell: dict[str, list[str]],
             thresholds: dict,
             full_cell_set: bool) -> Verdict:
    comps = thresholds["comparisons"]
    failures: list[str] = []
    warnings_: list[str] = []
    rows: list[tuple[str, float, float, float]] = []
    n_checked = 0

    for cell_id, viols in violations_by_cell.items():
        for v in viols:
            failures.append(f"[structural] {cell_id}: {v}")

    fam_z: dict[str, list[float]] = {}
    for key, spec in comps.items():
        cell_id, group, metric = key.split("|")
        if cell_id not in scores_by_cell:
            continue
        n_checked += 1
        value = scores_by_cell[cell_id].get(f"{group}|{metric}", float("nan"))
        thr = spec["threshold"]
        ratio = value / thr if thr > 0 and math.isfinite(value) else float("nan")
        rows.append((key, value, thr, ratio))
        if math.isnan(value):
            failures.append(f"[metric] {key}: not computable on candidate pool "
                            f"(reference produces a finite value)")
        elif value > thr:
            failures.append(f"[metric] {key}: {value:.4g} > threshold {thr:.4g}")
        if spec.get("null_std", 0) and spec["null_std"] > 0 and math.isfinite(value):
            fam_z.setdefault(metric, []).append(
                (value - spec["null_mean"]) / spec["null_std"])

    aggregates: dict[str, dict[str, float]] = {}
    agg_spec = thresholds.get("aggregates", {})
    if full_cell_set:
        for metric, spec in agg_spec.items():
            zs = fam_z.get(metric, [])
            if not zs:
                continue
            z = float(np.mean(zs))
            aggregates[metric] = {"z": z, "threshold": spec["threshold"],
                                  "n_terms": len(zs)}
            if z > spec["threshold"]:
                failures.append(f"[aggregate] {metric}: mean z {z:.3f} > "
                                f"threshold {spec['threshold']:.3f}")
    else:
        warnings_.append("partial cell set — aggregate family checks skipped")

    return Verdict(passed=not failures, failures=failures, warnings=warnings_,
                   n_gated=len(comps), n_checked=n_checked,
                   aggregates=aggregates, rows=rows)


def print_verdict(v: Verdict) -> None:
    rows = sorted((r for r in v.rows if math.isfinite(r[3])),
                  key=lambda r: -r[3])
    click.echo(f"\n{'comparison':<48}{'value':>12}{'threshold':>12}{'margin':>9}")
    for key, value, thr, ratio in rows[:15]:
        click.echo(f"{key:<48}{value:>12.4g}{thr:>12.4g}{ratio:>9.2f}")
    if len(rows) > 15:
        click.echo(f"  … {len(rows) - 15} more (all below threshold)"
                   if v.passed else f"  … {len(rows) - 15} more")
    for metric, a in v.aggregates.items():
        flag = "" if a["z"] <= a["threshold"] else "  ← FAIL"
        click.echo(f"aggregate {metric:<24} mean z {a['z']:>7.3f}  "
                   f"(threshold {a['threshold']:.3f}, {a['n_terms']} terms){flag}")
    for w in v.warnings:
        click.echo(f"WARNING: {w}")
    if v.failures:
        click.echo(f"\n{len(v.failures)} failure(s):")
        for f in v.failures:
            click.echo(f"  {f}")
    click.echo(f"\nTier 2(b) verdict: {'EQUIVALENT (PASS)' if v.passed else 'DIFFERENT (FAIL)'}"
               f"  [{v.n_checked}/{v.n_gated} gated comparisons checked]")


# ---------------------------------------------------------------------------
# Startup self-checks
# ---------------------------------------------------------------------------

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def startup_selfcheck(profile, profile_sha: str, manifest: dict) -> None:
    """Prove the local reference engine is on-contract before trusting it."""
    recorded = manifest["_meta"]["profile_sha256"]
    if profile_sha != recorded:
        raise click.ClickException(
            f"bundled profile sha {profile_sha[:12]}… != manifest's {recorded[:12]}… — "
            "the golden contract does not match this working tree.")
    by_id = {e["id"]: e for e in manifest["vectors"]}
    entry = by_id["f1_default"]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "f1.mid"
        humanise(GOLDEN_DIR / entry["input"], out, profile, **entry["params"])
        if _sha256_bytes(out.read_bytes()) != entry["sha256_output"]:
            raise click.ClickException(
                "reference engine failed to reproduce golden vector f1_default — "
                "run scripts/verify_golden.py; the local engine is off-contract "
                "and cannot serve as the Tier 2 reference.")
    if not FULL_KIT_FIXTURE.exists():
        raise click.ClickException(
            f"{FULL_KIT_FIXTURE} missing — run: python scripts/compare_port.py fixture")
    if FULL_KIT_FIXTURE.read_bytes() != full_kit_bytes():
        raise click.ClickException(
            f"{FULL_KIT_FIXTURE} does not match its builder — the checked-in "
            "fixture was modified; regenerate via: python scripts/compare_port.py fixture")


def _score_candidate_cells(
    cells: list[Cell], profile,
    candidate_runs: "dict[str, list[RunResult]]",
    identity_files: "dict[str, list[Path]]",
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    """Shared core of verify/self-null: reference pools + scoring + structural."""
    scores: dict[str, dict[str, float]] = {}
    violations: dict[str, list[str]] = {}
    for cell in cells:
        if cell.kind == "identity":
            viols: list[str] = []
            for p in identity_files.get(cell.id, []):
                vs, ws = check_identity(cell.input_path, p)
                viols += [f"{p.name}: {v}" for v in vs]
                for w in ws:
                    click.echo(f"  note [{cell.id}] {p.name}: {w}")
            if viols:
                violations[cell.id] = viols
            continue
        cd = prepare_cell(cell)
        runs_c = candidate_runs[cell.id]
        run_viols = [v for r in runs_c for v in r.violations]
        if run_viols:
            violations[cell.id] = run_viols
            continue
        ref_seeds = [REFERENCE_SEED_BASE + i for i in range(K_RUNS)]
        runs_ref = generate_runs(cd, profile, ref_seeds)
        scores[cell.id] = score_pools(cd, runs_c, runs_ref)
        click.echo(f"  scored {cell.id} ({len(cd.notes)} hits × {K_RUNS} runs/side)")
    return scores, violations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Tier 2 port-vs-reference equivalence runner (porting contract Tier 2(b))."""


def _filter_cells(cells: list[Cell], subset: str | None) -> tuple[list[Cell], bool]:
    if not subset:
        return cells, True
    wanted = {s.strip() for s in subset.split(",") if s.strip()}
    unknown = wanted - {c.id for c in cells}
    if unknown:
        raise click.ClickException(f"unknown cell id(s): {sorted(unknown)}")
    return [c for c in cells if c.id in wanted], False


def _finish(scores, violations, thresholds_path: Path, full_set: bool,
            json_out: Path | None) -> None:
    thresholds = load_thresholds(thresholds_path)
    if thresholds is None:
        click.echo(f"\nNo thresholds file at {thresholds_path} — metrics are "
                   "informational only (calibration not yet run/locked).")
        for cell_id, sc in scores.items():
            worst = sorted(sc.items(), key=lambda kv: -(kv[1] if math.isfinite(kv[1]) else -1))
            click.echo(f"  {cell_id}: " + ", ".join(
                f"{k}={v:.3g}" for k, v in worst[:4]))
        any_viol = any(violations.values())
        for cell_id, viols in violations.items():
            for v in viols:
                click.echo(f"  STRUCTURAL {cell_id}: {v}")
        sys.exit(1 if any_viol else 0)
    verdict = evaluate(scores, violations, thresholds, full_set)
    print_verdict(verdict)
    if json_out is not None:
        payload = {
            "verdict": "pass" if verdict.passed else "fail",
            "failures": verdict.failures,
            "aggregates": verdict.aggregates,
            "scores": scores,
            "structural_violations": violations,
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=1))
        click.echo(f"report written to {json_out}")
    sys.exit(0 if verdict.passed else 1)


@cli.command()
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False,
                                                 path_type=Path))
@click.option("--thresholds", type=click.Path(path_type=Path),
              default=THRESHOLDS_DEFAULT, show_default=True)
@click.option("--cells", "subset", default=None,
              help="Comma-separated cell ids: partial verification during porting "
                   "(aggregate family checks are skipped).")
@click.option("--json-out", type=click.Path(path_type=Path), default=None)
def verify(candidate_dir: Path, thresholds: Path, subset: str | None,
           json_out: Path | None) -> None:
    """Grade a port's candidate outputs against the reference engine."""
    manifest = load_manifest()
    profile, profile_sha = load_bundled_profile()
    startup_selfcheck(profile, profile_sha, manifest)
    cells, full_set = _filter_cells(load_cells(manifest), subset)

    candidate_runs: dict[str, list[RunResult]] = {}
    identity_files: dict[str, list[Path]] = {}
    for cell in cells:
        cell_dir = candidate_dir / cell.id
        files = sorted(cell_dir.glob("*.mid")) if cell_dir.is_dir() else []
        if cell.kind == "identity":
            if not files:
                raise click.ClickException(
                    f"missing candidate outputs for identity cell {cell.id} "
                    f"(need >= 1 file in {cell_dir})")
            identity_files[cell.id] = files
            continue
        if len(files) != K_RUNS:
            raise click.ClickException(
                f"cell {cell.id}: found {len(files)} candidate files in {cell_dir}, "
                f"need exactly {K_RUNS} (one per independent-seed run — the "
                "thresholds are calibrated at this pool size)")
        cd = prepare_cell(cell)
        candidate_runs[cell.id] = load_candidate_runs(cd, files)

    click.echo(f"Scoring {len(cells)} cells (K={K_RUNS} runs per side)…")
    scores, violations = _score_candidate_cells(cells, profile,
                                                candidate_runs, identity_files)
    _finish(scores, violations, thresholds, full_set, json_out)


@cli.command("self-null")
@click.option("--cells", "subset", default=None,
              help="Comma-separated cell ids (default: all).")
@click.option("--candidate-seed-base", default=888_000, show_default=True,
              help="Seed base for the in-process 'candidate' pool (disjoint from "
                   "the pinned reference base).")
@click.option("--thresholds", type=click.Path(path_type=Path),
              default=THRESHOLDS_DEFAULT, show_default=True)
@click.option("--json-out", type=click.Path(path_type=Path), default=None)
def self_null(subset: str | None, candidate_seed_base: int, thresholds: Path,
              json_out: Path | None) -> None:
    """Reference vs itself at disjoint seeds — must be EQUIVALENT.

    The runner's own smoke test (and the CI lock): a correct 'port' — the
    reference engine — must pass the locked thresholds."""
    manifest = load_manifest()
    profile, profile_sha = load_bundled_profile()
    startup_selfcheck(profile, profile_sha, manifest)
    cells, full_set = _filter_cells(load_cells(manifest), subset)
    if candidate_seed_base == REFERENCE_SEED_BASE:
        raise click.ClickException("candidate seed base must differ from the "
                                   f"pinned reference base {REFERENCE_SEED_BASE}")

    candidate_runs: dict[str, list[RunResult]] = {}
    identity_files: dict[str, list[Path]] = {}
    tmp = tempfile.TemporaryDirectory()
    tmp_dir = Path(tmp.name)
    for cell in cells:
        if cell.kind == "identity":
            out = tmp_dir / f"{cell.id}.mid"
            humanise(cell.input_path, out, profile, seed=candidate_seed_base,
                     **cell.params)
            identity_files[cell.id] = [out]
            continue
        cd = prepare_cell(cell)
        seeds = [candidate_seed_base + i for i in range(K_RUNS)]
        candidate_runs[cell.id] = generate_runs(cd, profile, seeds)

    click.echo(f"Self-null over {len(cells)} cells (K={K_RUNS} per side)…")
    scores, violations = _score_candidate_cells(cells, profile,
                                                candidate_runs, identity_files)
    tmp.cleanup()
    _finish(scores, violations, thresholds, full_set, json_out)


@cli.command()
@click.option("--check", is_flag=True, help="Verify the checked-in fixture matches "
                                            "the builder instead of writing it.")
def fixture(check: bool) -> None:
    """Write (or verify) the runner-owned full-kit coverage fixture."""
    data = full_kit_bytes()
    if check:
        if not FULL_KIT_FIXTURE.exists():
            raise click.ClickException(f"{FULL_KIT_FIXTURE} does not exist")
        if FULL_KIT_FIXTURE.read_bytes() != data:
            raise click.ClickException(f"{FULL_KIT_FIXTURE} differs from the builder")
        click.echo(f"ok: {FULL_KIT_FIXTURE} matches its builder "
                   f"(sha256 {_sha256_bytes(data)[:12]}…)")
        return
    FULL_KIT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FULL_KIT_FIXTURE.write_bytes(data)
    click.echo(f"wrote {FULL_KIT_FIXTURE} (sha256 {_sha256_bytes(data)[:12]}…)")


@cli.command("dump-reference")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
def dump_reference(out_dir: Path) -> None:
    """Write the reference pools (K runs per cell) for port developers."""
    manifest = load_manifest()
    profile, profile_sha = load_bundled_profile()
    startup_selfcheck(profile, profile_sha, manifest)
    cells = load_cells(manifest)
    for cell in cells:
        cell_dir = out_dir / cell.id
        cell_dir.mkdir(parents=True, exist_ok=True)
        if cell.kind == "identity":
            out = cell_dir / "ref_identity.mid"
            humanise(cell.input_path, out, profile, seed=REFERENCE_SEED_BASE,
                     **cell.params)
        else:
            for i in range(K_RUNS):
                humanise(cell.input_path, cell_dir / f"ref_{i:03d}.mid", profile,
                         seed=REFERENCE_SEED_BASE + i, **cell.params)
        click.echo(f"  {cell.id}")
    (out_dir / "README.txt").write_text(
        "Reference-engine output pools for Tier 2(b), one directory per cell.\n"
        f"Generated with seeds {REFERENCE_SEED_BASE}+i (the runner's pinned pool).\n"
        "A port must supply its OWN pools in the same layout (seeds free):\n"
        f"exactly {K_RUNS} .mid files per distributional cell, >=1 per identity "
        "cell,\nthen run: python scripts/compare_port.py verify <dir>\n")
    click.echo(f"reference pools written to {out_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cli()
