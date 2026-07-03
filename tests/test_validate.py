"""Tests for scripts/validate.py — the Part C validation harness."""

import sys
from pathlib import Path
from types import SimpleNamespace

import mido
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from validate import (
    MIN_EVAL_HITS,
    Take,
    build_programmed_midi,
    coarsen_velocities,
    evaluate_level,
    load_take,
    programmed_velocities,
    read_output_hits,
    role_labels,
    signed_offset_ms,
)

from pocketmidi.humanise import load_profile
from pocketmidi.midi_utils import build_tempo_map, quantise_to_grid

SHIPPED_PROFILE = Path(__file__).parent.parent / "pocketmidi" / "profiles" / "rock.json"


# ── coarsen_velocities ───────────────────────────────────────────────────────

def test_coarsen_flat_maps_to_median():
    v = np.array([10, 20, 30, 40, 100])
    out = coarsen_velocities(v, "flat")
    assert np.all(out == 30)


def test_coarsen_is_monotone_and_tie_preserving():
    rng = np.random.RandomState(0)
    v = rng.randint(1, 128, size=200).astype(float)
    v[10] = v[20] = v[30] = 77.0   # forced ties
    out = coarsen_velocities(v, 4)
    # ≤ 4 distinct palette values
    assert len(np.unique(out)) <= 4
    # ties preserved
    assert out[10] == out[20] == out[30]
    # monotone: lower input velocity never maps to a higher palette value
    order = np.argsort(v, kind="stable")
    assert np.all(np.diff(out[order]) >= 0)


def test_coarsen_constant_input():
    v = np.full(16, 64.0)
    out = coarsen_velocities(v, 4)
    assert np.all(out == 64)


def test_coarsen_two_level():
    v = np.array([20.0] * 10 + [100.0] * 10)   # ghost/accent two-cluster
    out = coarsen_velocities(v, 2)
    assert np.all(out[:10] == 20)
    assert np.all(out[10:] == 100)


# ── programmed input construction ────────────────────────────────────────────

def _synthetic_take(ppq=480, tempo=500_000, bars=4):
    """Kick 1&3, snare 2&4, 8th hats — jittered timing, varied velocity."""
    rng = np.random.RandomState(42)
    recs = []
    for bar in range(bars):
        bar_tick = bar * 4 * ppq
        for beat, note, vel in [(0, 36, 100), (2, 36, 98), (1, 38, 95), (3, 38, 96)]:
            recs.append((bar_tick + beat * ppq + rng.randint(-20, 21), note,
                         vel + rng.randint(-6, 7)))
        for e in range(8):
            recs.append((bar_tick + e * ppq // 2 + rng.randint(-15, 16), 42,
                         70 + rng.randint(-15, 16)))
    recs = [(max(0, t), n, v) for t, n, v in recs]
    recs.sort(key=lambda r: (quantise_to_grid(r[0], ppq), r[0], r[1]))
    grid = np.array([quantise_to_grid(t, ppq) for t, _, _ in recs])
    abst = np.array([t for t, _, _ in recs])
    notes = np.array([n for _, n, _ in recs])
    vels = np.array([v for _, _, v in recs])
    tempo_map = [(0, tempo)]
    from pocketmidi.midi_utils import TD11_TO_GROUP, grid_position_in_bar, ticks_to_ms_with_map
    return Take(
        take_id="synthetic/take1", beat_type="beat", bpm=120.0, ppq=ppq,
        tempo_map=tempo_map, notes=notes,
        groups=np.array([TD11_TO_GROUP[n] for n in notes]),
        grid_ticks=grid,
        grid_pos=np.array([grid_position_in_bar(g, ppq) for g in grid]),
        human_abs=abst,
        human_off=np.array([signed_offset_ms(g, a, tempo_map, ppq)
                            for g, a in zip(grid, abst)]),
        human_vel=vels,
        human_t=np.array([ticks_to_ms_with_map(0, a, tempo_map, ppq) for a in abst]),
    )


def test_programmed_midi_roundtrip(tmp_path):
    take = _synthetic_take()
    pv = programmed_velocities(take, 4)
    mid = build_programmed_midi(take, pv)
    p = tmp_path / "prog.mid"
    mid.save(str(p))

    reread = mido.MidiFile(str(p))
    notes, ticks, vels = [], [], []
    for track in reread.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                assert msg.channel == 9
                notes.append(msg.note)
                ticks.append(abs_tick)
                vels.append(msg.velocity)
    assert np.array_equal(np.array(notes), take.notes)
    assert np.array_equal(np.array(ticks), take.grid_ticks)   # exactly on grid
    assert np.array_equal(np.array(vels), pv)
    assert build_tempo_map(reread) == take.tempo_map


def test_programmed_velocities_per_instrument_palette():
    take = _synthetic_take()
    pv = programmed_velocities(take, 4)
    for g in np.unique(take.groups):
        assert len(np.unique(pv[take.groups == g])) <= 4


def test_signed_offset_ms_sign_convention():
    tempo_map = [(0, 500_000)]   # 120 BPM → 1 tick at ppq=480 is 500/480 ms
    assert signed_offset_ms(480, 528, tempo_map, 480) == pytest.approx(50.0)
    assert signed_offset_ms(480, 432, tempo_map, 480) == pytest.approx(-50.0)
    assert signed_offset_ms(480, 480, tempo_map, 480) == 0.0


# ── load_take ────────────────────────────────────────────────────────────────

def _write_human_midi(path, take):
    mid = mido.MidiFile(type=0, ticks_per_beat=take.ppq)
    tr = mido.MidiTrack()
    events = [(0, mido.MetaMessage("set_tempo", tempo=take.tempo_map[0][1], time=0))]
    order = np.argsort(take.human_abs, kind="stable")
    for i in order:
        events.append((int(take.human_abs[i]),
                       mido.Message("note_on", note=int(take.notes[i]),
                                    velocity=int(take.human_vel[i]), channel=9, time=0)))
    prev = 0
    for tick, msg in sorted(events, key=lambda e: e[0]):
        tr.append(msg.copy(time=tick - prev))
        prev = tick
    tr.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(tr)
    mid.save(str(path))


def test_load_take_roundtrip(tmp_path):
    take = _synthetic_take()
    _write_human_midi(tmp_path / "h.mid", take)
    row = SimpleNamespace(midi_filename="h.mid", id="synthetic/take1",
                          beat_type="beat", bpm=120)
    loaded = load_take(tmp_path, row)
    assert loaded is not None
    assert np.array_equal(loaded.notes, take.notes)
    assert np.array_equal(loaded.grid_ticks, take.grid_ticks)
    assert np.array_equal(loaded.human_vel, take.human_vel)
    np.testing.assert_allclose(loaded.human_off, take.human_off)


def test_load_take_rejects_tiny_takes(tmp_path):
    take = _synthetic_take(bars=1)
    assert len(take.notes) < MIN_EVAL_HITS
    _write_human_midi(tmp_path / "h.mid", take)
    row = SimpleNamespace(midi_filename="h.mid", id="x", beat_type="beat", bpm=120)
    assert load_take(tmp_path, row) is None


# ── end-to-end smoke ─────────────────────────────────────────────────────────

def test_evaluate_level_smoke(tmp_path):
    """Whole pipeline on one synthetic take with the shipped profile:
    aligned output, sane metric structure, input baseline exactly on-grid."""
    take = _synthetic_take(bars=8)
    prof = load_profile(SHIPPED_PROFILE)
    results, groups = evaluate_level(
        [take], 4, {"shipped": prof}, seeds=[1, 2], workdir=tmp_path
    )

    # one small take stays under MIN_GROUP_HITS — only the ALL row is emitted
    assert groups == []
    human = results["human"]["ALL"]
    inp = results["input"]["ALL"]
    out_runs = results["profiles"]["shipped"]
    assert len(out_runs) == 2

    # human reference has real spread; input is exactly on-grid and identical to itself
    assert human["off_sigma"] > 0
    assert inp["off_sigma"] == 0.0
    assert inp["off_mean"] == 0.0
    assert inp["spear_in"] == pytest.approx(1.0)
    assert inp["contour_mae"] > 0   # coarsening error vs original

    for run in out_runs:
        r = run["ALL"]
        # humanised output moved off the grid, with finite distances
        assert r["off_sigma"] > 0
        assert np.isfinite(r["off_w1"])
        assert np.isfinite(r["vjump_mean"])
        assert 0 <= r["off_ks"] <= 1

    # different seeds → different realisations
    assert out_runs[0]["ALL"]["off_mean"] != out_runs[1]["ALL"]["off_mean"]


# ── anti-robotic metrics (spec addendum Fix 2) ───────────────────────────────

def test_role_labels_two_cluster_ghost_accent():
    v = [30] * 8 + [100] * 8
    roles = role_labels(v)
    assert set(roles[:8]) == {"soft"}
    assert set(roles[8:]) == {"hard"}


def test_role_labels_single_role_without_evidence():
    # constant velocities → engine falls back → one shared role label
    roles = role_labels([90] * 20)
    assert len(set(roles)) == 1


def test_role_labels_fixed_across_conditions():
    # labels derive from the HUMAN velocities only — they are a property of the
    # hit, not of any condition's output
    v = [30, 100] * 8
    assert list(role_labels(v)) == list(role_labels(v))


def test_antirobotic_metrics_flat_vs_human(tmp_path):
    """Flat input must score worst: zero-jump mass 1.0 and within-role sigma 0.
    The human reference must show real micro-variation on both."""
    take = _synthetic_take(bars=8)
    from pocketmidi.humanise import load_profile as _lp
    prof = _lp(SHIPPED_PROFILE)
    results, _ = evaluate_level([take], "flat", {"shipped": prof}, seeds=[1], workdir=tmp_path)

    inp = results["input"]["ALL"]
    hum = results["human"]["ALL"]
    out = results["profiles"]["shipped"][0]["ALL"]

    assert inp["zjump_mass"] == 1.0          # every adjacent delta is exactly 0
    assert inp["wrole_sigma"] == 0.0         # no spread within any role
    assert hum["zjump_mass"] < 0.5           # humans rarely repeat identical velocities
    assert hum["wrole_sigma"] > 0
    # humanised output restores variation: strictly less robotic than the flat input
    assert out["zjump_mass"] < inp["zjump_mass"]
    assert out["wrole_sigma"] > 0


def test_antirobotic_zero_jump_mass_coarsened_input(tmp_path):
    """4-level coarsening collapses micro-differences into identical repeated
    values → its zero-jump mass must sit clearly above the human original's.
    (The within-role direction needs real role-structured playing — that is
    what the real-GMD behaviour table validates; here we lock the mechanics.)"""
    take = _synthetic_take(bars=8)
    from pocketmidi.humanise import load_profile as _lp
    prof = _lp(SHIPPED_PROFILE)
    results, _ = evaluate_level([take], 4, {"shipped": prof}, seeds=[1], workdir=tmp_path)
    assert results["input"]["ALL"]["zjump_mass"] > results["human"]["ALL"]["zjump_mass"] * 2


def test_wrole_sigma_math():
    """Exact mechanics: mean of per-(take, group, pos, role) stds over cells with
    >= MIN_CELL_N hits; roles slice positions into separate cells."""
    import pandas as pd
    from validate import _wrole_sigma
    df = pd.DataFrame({
        "take": ["t"] * 12,
        "group": ["snare"] * 12,
        "pos": [4] * 12,
        # one position, two roles: ghosts spread {28,30,32}x2, accents constant 100
        "role": ["soft"] * 6 + ["hard"] * 6,
        "vel": [28.0, 30.0, 32.0, 28.0, 30.0, 32.0] + [100.0] * 6,
    })
    expected_soft = np.std([28, 30, 32, 28, 30, 32], ddof=1)
    assert _wrole_sigma(df) == pytest.approx((expected_soft + 0.0) / 2)
    # without the role dimension the blended cell would show a huge fake spread
    df_norole = df.assign(role="all")
    assert _wrole_sigma(df_norole) > 30


def test_read_output_hits_detects_misalignment(tmp_path):
    take = _synthetic_take()
    pv = programmed_velocities(take, 4)
    mid = build_programmed_midi(take, pv)
    p = tmp_path / "prog.mid"
    mid.save(str(p))
    # a file with one hit missing must be rejected
    shorter = _synthetic_take()
    shorter.notes[0] = 57  # crash instead of the expected first note
    with pytest.raises(RuntimeError, match="misalignment"):
        read_output_hits(p, shorter)
