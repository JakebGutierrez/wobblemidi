"""Validation harness for the module-13 velocity rebuild (rebuild spec Part C).

Measures distance-to-human on held-out GMD rock takes. For each held-out human
performance a "programmed" input is built — timing quantised to the 16th grid,
velocity coarsened to a small per-instrument palette per take (a producer
programs a contour, not micro-noise) — pocketmidi is run on it, and the output
is compared against the real human original.

Conditions:
  human     — the original performance (reference values, not a condition)
  input     — the programmed input itself, unprocessed        (baseline a)
  gate      — old-schema profile built from the GMD *train*
              split only                                       (baseline b — THE GATE)
  shipped   — pocketmidi/profiles/rock.json (baseline c — context only: it was
              built from ALL rock takes, so it has seen the held-out takes)
  candidate — optional rebuilt-schema profile (--candidate; checkpoint 3)

Split: GMD's own take-level `split` column. The gate profile is built from
split=="train" only; evaluation runs on split=="test". The `validation` split
is left unused this round.

Metric conventions (all per instrument group, pooled over held-out takes):
  * offsets are signed ms from the hit's programmed 16th-grid slot (the human
    reference uses its own quantised slot) — same math as build_profiles.py.
  * lag-1 autocorrelations follow calibrate_phi.py: residuals are de-meaned per
    (instrument, grid-position) within the condition, same-slot hits collapsed
    by mean; "ALL" collapses kit-wide across instruments (timing) while
    per-instrument rows keep sequences within that instrument. Velocity lag-1
    uses per-instrument sequences de-meaned per (take, instrument, position) —
    within-take fine structure only, the audit's / spec B2's convention.
  * every condition's hits align 1:1 with the input hits (humanise preserves
    note count and order), so contour metrics are hit-matched, not re-paired.
  * anti-robotic metrics (spec addendum Fix 2): zero micro-jump mass (fraction of
    adjacent same-instrument velocity deltas with |dv| <= 1 — a coarsened input
    repeats identical values, humans almost never do) and within-(position, role)
    velocity sigma (spread within a fixed musical role — ~0 is robotic, excess is
    noise; scored TWO-SIDED against the human reference, not against the input).
    Role labels are FIXED per hit: derived once from the HUMAN original velocities
    using the engine's own relative-tier convention (_file_tier_thresholds), so
    every condition is measured on identical cells.

Robustness: profile conditions run over multiple fixed seeds; the report shows
mean ± 95% CI half-width across seeds. Each take gets a distinct per-take seed
derived from the run seed so takes do not share an RNG stream.

Usage:
    python scripts/validate.py <path/to/groove> [--candidate rebuilt.json] [--sensitivity]
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import tempfile
from pathlib import Path

import click
import mido
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, t as t_dist, wasserstein_distance

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pocketmidi.humanise import (
    LoadedProfile,
    _file_tier_thresholds,
    _velocity_tier,
    humanise,
    load_profile,
)
from pocketmidi.midi_utils import (
    TD11_TO_GROUP,
    build_tempo_map,
    grid_position_in_bar,
    is_four_four,
    quantise_to_grid,
    ticks_to_ms_with_map,
)

from build_profiles import build_profile_output, collect_hits

DEFAULT_SEEDS = (101, 202, 303, 404, 505)
MIN_EVAL_HITS = 20      # takes with fewer extracted hits are excluded (tiny fills)
MIN_CELL_N = 3          # min hits in a (take, group, pos) cell for within-position sigma
MIN_SPEARMAN_N = 8      # min hits in a (take, group) cell for a Spearman value
MIN_PAIRS = 30          # min pooled pairs for a lag-1 / gap-sigma value
MIN_GROUP_HITS = 200    # min pooled human hits for a per-instrument report row
GROUP_ORDER = [
    "kick", "snare", "hihat_closed", "hihat_open", "ride", "crash",
    "tom_high", "tom_mid", "tom_low",
]
GATE_PROFILE_DEFAULT = REPO_ROOT / "validation" / "gate_old_schema.json"
SHIPPED_PROFILE = REPO_ROOT / "pocketmidi" / "profiles" / "rock.json"
REPORT_DEFAULT = REPO_ROOT / "validation" / "last_report.json"

# (key, label, kind) — kind "prop": human column shows the reference value;
# "dist": a distance-to-human (human column is trivially 0); "corr": a
# correlation against a reference contour (human column trivially 1 for
# spear_orig). xgap_sigma only exists at the ALL level.
METRIC_ROWS = [
    ("off_mean",       "timing offset mean [ms]",              "prop"),
    ("off_sigma",      "timing offset sigma [ms]",             "prop"),
    ("off_w1",         "offset W1 dist-to-human [ms]",         "dist"),
    ("off_ks",         "offset KS dist-to-human",              "dist"),
    ("t_lag1",         "timing lag-1 autocorr",                "prop"),
    ("xgap_sigma",     "same-slot cross-instr gap sigma [ms]", "prop"),
    ("vel_wpos_sigma", "velocity within-position sigma",       "prop"),
    ("zjump_mass",     "zero micro-jump mass (|dv|<=1)",       "prop"),
    ("wrole_sigma",    "within-(position, role) vel sigma",    "prop"),
    ("vjump_mean",     "adjacent velocity-jump mean",          "prop"),
    ("vjump_w1",       "velocity-jump W1 dist-to-human",       "dist"),
    ("v_lag1",         "velocity lag-1 autocorr",              "prop"),
    ("contour_mae",    "contour MAE vs original [vel]",        "dist"),
    ("spear_in",       "Spearman vs programmed input",         "corr"),
    ("spear_orig",     "Spearman vs original",                 "corr"),
]


# ---------------------------------------------------------------------------
# Take loading and programmed-input construction
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Take:
    """One held-out human performance, as aligned per-hit arrays.

    Canonical hit order is sorted by (grid_tick, abs_tick, note); the programmed
    input writes note_ons in this order and humanise preserves it, so every
    condition's hit i corresponds to human hit i.
    """
    take_id: str
    beat_type: str
    bpm: float
    ppq: int
    tempo_map: list[tuple[int, int]]
    notes: np.ndarray       # (N,) int
    groups: np.ndarray      # (N,) str
    grid_ticks: np.ndarray  # (N,) int
    grid_pos: np.ndarray    # (N,) int
    human_abs: np.ndarray   # (N,) int
    human_off: np.ndarray   # (N,) float ms
    human_vel: np.ndarray   # (N,) int
    human_t: np.ndarray     # (N,) float — absolute ms position of the human hit


def signed_offset_ms(grid_tick: int, abs_tick: int, tempo_map, ppq: int) -> float:
    """Signed ms from grid slot to hit: positive = late (same math as build_profiles)."""
    if abs_tick >= grid_tick:
        return ticks_to_ms_with_map(grid_tick, abs_tick, tempo_map, ppq)
    return -ticks_to_ms_with_map(abs_tick, grid_tick, tempo_map, ppq)


def load_take(gmd_dir: Path, row) -> Take | None:
    """Load one info.csv row as a Take, or None if unusable (missing/unreadable/
    non-4/4/too few hits). Keeps ALL hits — no ghost-note floor: the programmed
    input must contain the ghosts a producer would program."""
    midi_path = gmd_dir / row.midi_filename
    if not midi_path.exists():
        return None
    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception:
        return None
    if not is_four_four(mid):
        return None

    ppq = mid.ticks_per_beat
    tempo_map = build_tempo_map(mid)

    recs: list[tuple[int, int, int, int]] = []   # (grid_tick, abs_tick, note, velocity)
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0 and msg.note in TD11_TO_GROUP:
                recs.append((quantise_to_grid(abs_tick, ppq), abs_tick, msg.note, msg.velocity))
    if len(recs) < MIN_EVAL_HITS:
        return None

    recs.sort(key=lambda r: (r[0], r[1], r[2]))
    grid_ticks = np.array([r[0] for r in recs], dtype=int)
    human_abs = np.array([r[1] for r in recs], dtype=int)
    notes = np.array([r[2] for r in recs], dtype=int)
    vels = np.array([r[3] for r in recs], dtype=int)

    return Take(
        take_id=str(row.id),
        beat_type=str(row.beat_type),
        bpm=float(row.bpm),
        ppq=ppq,
        tempo_map=tempo_map,
        notes=notes,
        groups=np.array([TD11_TO_GROUP[n] for n in notes]),
        grid_ticks=grid_ticks,
        grid_pos=np.array([grid_position_in_bar(g, ppq) for g in grid_ticks], dtype=int),
        human_abs=human_abs,
        human_off=np.array([
            signed_offset_ms(g, a, tempo_map, ppq) for g, a in zip(grid_ticks, human_abs)
        ]),
        human_vel=vels,
        human_t=np.array([ticks_to_ms_with_map(0, a, tempo_map, ppq) for a in human_abs]),
    )


def coarsen_velocities(vels: np.ndarray, levels: int | str) -> np.ndarray:
    """Map velocities to a small palette: quantile bins, each replaced by its bin
    mean ("flat" = everything to the median). Monotone (contour ordering survives
    at palette granularity) and tie-preserving (equal in → equal out)."""
    v = np.asarray(vels, dtype=float)
    if levels == "flat":
        return np.full(len(v), int(np.clip(round(float(np.median(v))), 1, 127)), dtype=int)
    k = int(levels)
    edges = np.quantile(v, np.linspace(0.0, 1.0, k + 1)[1:-1])
    bins = np.searchsorted(edges, v, side="right")
    out = np.empty(len(v), dtype=int)
    for b in np.unique(bins):
        m = bins == b
        out[m] = int(np.clip(round(float(v[m].mean())), 1, 127))
    return out


def programmed_velocities(take: Take, levels: int | str) -> np.ndarray:
    """Per-instrument palette per take (spec O1)."""
    out = np.empty(len(take.notes), dtype=int)
    for g in np.unique(take.groups):
        m = take.groups == g
        out[m] = coarsen_velocities(take.human_vel[m], levels)
    return out


_ROLE_SENTINEL = (-1.0, -1.0)


def role_labels(human_vels) -> np.ndarray:
    """FIXED per-hit role labels for one (take, instrument) from the HUMAN original.

    Uses the engine's relative-tier convention (_file_tier_thresholds / B4) so the
    harness, build, and runtime all share one notion of "role". Labels come from the
    human take — not from each condition's own velocities — so every condition is
    compared on identical (position, role) cells (the Codex fixed-labels tightening),
    and roles stay meaningful even for a flat input. Insufficient evidence (the
    engine would fall back to absolute thresholds, which are not meaningful here)
    → one single role, degrading the cell to plain (position)."""
    v = np.asarray(human_vels, dtype=float)
    low, high = _file_tier_thresholds(v, _ROLE_SENTINEL)
    if (low, high) == _ROLE_SENTINEL:
        return np.full(len(v), "all", dtype=object)
    return np.array([_velocity_tier(x, (low, high)) for x in v], dtype=object)


def take_role_labels(take: Take) -> np.ndarray:
    roles = np.empty(len(take.notes), dtype=object)
    for g in np.unique(take.groups):
        m = take.groups == g
        roles[m] = role_labels(take.human_vel[m])
    return roles


def build_programmed_midi(take: Take, prog_vels: np.ndarray) -> mido.MidiFile:
    """Single-track type-0 file: original tempo map, 4/4, note_ons on the grid.

    Deliberately writes NO note_offs: in humanise() a note_off is a fixed event
    that both caps late shifts (paired-off / next_fixed ceiling) and, once
    interleaved between note_ons, floors early shifts via prev_emitted_abs — the
    harness must measure the engine's distributions, not note-length clamping.
    Only note_ons are measured. end_of_track is placed well past the last hit so
    it never clamps a late shift.
    """
    mid = mido.MidiFile(type=0, ticks_per_beat=take.ppq)
    track = mido.MidiTrack()

    events: list[tuple[int, int, object]] = [
        (0, 0, mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    ]
    for tick, tempo in take.tempo_map:
        events.append((tick, 0, mido.MetaMessage("set_tempo", tempo=tempo, time=0)))
    for i in range(len(take.notes)):
        events.append((
            int(take.grid_ticks[i]), 1,
            mido.Message("note_on", note=int(take.notes[i]),
                         velocity=int(prog_vels[i]), channel=9, time=0),
        ))
    end_tick = int(take.grid_ticks.max()) + 8 * take.ppq
    events.append((end_tick, 2, mido.MetaMessage("end_of_track", time=0)))

    events.sort(key=lambda e: (e[0], e[1]))   # stable: notes keep canonical order
    prev = 0
    for tick, _, msg in events:
        track.append(msg.copy(time=tick - prev))
        prev = tick
    mid.tracks.append(track)
    return mid


def read_output_hits(path: Path, take: Take) -> tuple[np.ndarray, np.ndarray]:
    """Extract (abs_ticks, velocities) of the output's drum note_ons, asserting
    1:1 alignment with the take's canonical hit order."""
    mid = mido.MidiFile(str(path))
    notes, abss, vels = [], [], []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0 and msg.note in TD11_TO_GROUP:
                notes.append(msg.note)
                abss.append(abs_tick)
                vels.append(msg.velocity)
    if len(notes) != len(take.notes) or not np.array_equal(np.array(notes), take.notes):
        raise RuntimeError(f"output/input hit misalignment for take {take.take_id}")
    return np.array(abss, dtype=int), np.array(vels, dtype=int)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _lag1(sub: pd.DataFrame, col: str, kitwide: bool,
          demean_keys: tuple[str, ...] = ("group", "pos")) -> float:
    """Pooled lag-1 autocorrelation of per-slot residuals of *col*.

    Timing uses pooled per-(group, pos) de-meaning — calibrate_phi.py's
    convention, which deliberately leaves the static take lean in the residual
    (that lean is part of what phi=0.4 represents while B3 is parked). Velocity
    passes ("take", "group", "pos") so the residual is within-take fine
    structure only — the audit's convention, and the thing B2's kick VelDrift
    actually changes; pooled de-meaning would let take-identity inflate it.
    """
    d = sub[["take", "group", "pos", "slot", col]].copy()
    d["resid"] = d[col] - d.groupby(list(demean_keys))[col].transform("mean")
    xs, ys = [], []
    keys = ["take"] if kitwide else ["take", "group"]
    for _, tsub in d.groupby(keys, sort=False):
        seq = tsub.groupby("slot", sort=True)["resid"].mean().to_numpy()
        if len(seq) >= 2:
            xs.append(seq[:-1])
            ys.append(seq[1:])
    if not xs:
        return float("nan")
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if len(x) < MIN_PAIRS or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _wpos_sigma(sub: pd.DataFrame) -> float:
    cells = sub.groupby(["take", "group", "pos"])["vel"].agg(["std", "count"])
    stds = cells.loc[cells["count"] >= MIN_CELL_N, "std"]
    return float(stds.mean()) if len(stds) else float("nan")


def _wrole_sigma(sub: pd.DataFrame) -> float:
    """Within-(position, role) velocity sigma (anti-robotic, addendum Fix 2).

    Role is the fixed human-derived label, so this is the spread a condition puts
    WITHIN one musical role at one position — ~0 means robotically identical hits,
    human-level means natural variation, far above human means noise."""
    cells = sub.groupby(["take", "group", "pos", "role"])["vel"].agg(["std", "count"])
    stds = cells.loc[cells["count"] >= MIN_CELL_N, "std"]
    return float(stds.mean()) if len(stds) else float("nan")


def _jumps(sub: pd.DataFrame, col: str) -> np.ndarray:
    """|velocity difference| between adjacent hits of the same instrument."""
    out = []
    d = sub.sort_values(["take", "group", "slot", "ord"], kind="stable")
    for _, tsub in d.groupby(["take", "group"], sort=False):
        v = tsub[col].to_numpy(dtype=float)
        if len(v) >= 2:
            out.append(np.abs(np.diff(v)))
    return np.concatenate(out) if out else np.array([])


def _contour_mae(sub: pd.DataFrame) -> float:
    """Hit-weighted MAE of per-(take, instrument, grid-position) mean velocity
    against the human original's per-cell mean."""
    cells = sub.groupby(["take", "group", "pos"]).agg(
        c=("vel", "mean"), h=("h_vel", "mean"), n=("vel", "size")
    )
    if not len(cells):
        return float("nan")
    return float(np.average(np.abs(cells["c"] - cells["h"]), weights=cells["n"]))


def _spearman(sub: pd.DataFrame, against: str) -> float:
    """Mean per-(take, instrument) Spearman rank corr of condition velocities
    against *against* (hit-matched)."""
    vals = []
    for _, tsub in sub.groupby(["take", "group"], sort=False):
        if len(tsub) < MIN_SPEARMAN_N:
            continue
        a = tsub["vel"].to_numpy(dtype=float)
        b = tsub[against].to_numpy(dtype=float)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        rho = spearmanr(a, b).statistic
        if not math.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")


def _xgap_sigma(sub: pd.DataFrame) -> float:
    """Sigma of signed ms gaps between same-slot hits of different instruments
    (pair order fixed by group name so signs are consistent across conditions)."""
    gaps = []
    for _, ssub in sub.groupby(["take", "slot"], sort=False):
        if len(ssub) < 2 or ssub["group"].nunique() < 2:
            continue
        rows = ssub.sort_values(["group", "ord"], kind="stable")
        tv = rows["t"].to_numpy()
        gr = rows["group"].to_numpy()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if gr[i] != gr[j]:
                    gaps.append(tv[i] - tv[j])
    if len(gaps) < MIN_PAIRS:
        return float("nan")
    return float(np.std(gaps, ddof=1))


def compute_metrics(
    df: pd.DataFrame, off: np.ndarray, vel: np.ndarray, t: np.ndarray, groups: list[str]
) -> dict[str, dict[str, float]]:
    """All Part C metrics for one condition realisation, per instrument + ALL."""
    d = df.copy()
    d["off"] = np.asarray(off, dtype=float)
    d["vel"] = np.asarray(vel, dtype=float)
    d["t"] = np.asarray(t, dtype=float)

    res: dict[str, dict[str, float]] = {}
    for g in ["ALL"] + groups:
        sub = d if g == "ALL" else d[d["group"] == g]
        jumps_c = _jumps(sub, "vel")
        jumps_h = _jumps(sub, "h_vel")
        r = {
            "off_mean": float(sub["off"].mean()),
            "off_sigma": float(sub["off"].std(ddof=1)),
            "off_w1": float(wasserstein_distance(sub["off"], sub["h_off"])),
            "off_ks": float(ks_2samp(sub["off"], sub["h_off"]).statistic),
            "t_lag1": _lag1(sub, "off", kitwide=(g == "ALL")),
            "vel_wpos_sigma": _wpos_sigma(sub),
            "zjump_mass": float((jumps_c <= 1.0).mean()) if len(jumps_c) else float("nan"),
            "wrole_sigma": _wrole_sigma(sub),
            "vjump_mean": float(np.mean(jumps_c)) if len(jumps_c) else float("nan"),
            "vjump_w1": (
                float(wasserstein_distance(jumps_c, jumps_h))
                if len(jumps_c) and len(jumps_h) else float("nan")
            ),
            "v_lag1": _lag1(sub, "vel", kitwide=False,
                            demean_keys=("take", "group", "pos")),
            "contour_mae": _contour_mae(sub),
            "spear_in": _spearman(sub, "i_vel"),
            "spear_orig": _spearman(sub, "h_vel"),
        }
        if g == "ALL":
            r["xgap_sigma"] = _xgap_sigma(sub)
        res[g] = r
    return res


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate_level(
    takes: list[Take],
    level: int | str,
    loaded_profiles: dict[str, LoadedProfile],
    seeds: list[int],
    workdir: Path,
) -> tuple[dict, list[str]]:
    """Run one input-coarseness level: build programmed inputs, run each profile
    over each seed, and compute metrics for human / input / every profile run.

    Returns (results, groups) where results = {
        "human": metrics, "input": metrics,
        "profiles": {name: [metrics per seed, ...]},
    }.
    """
    frames = []
    prog_paths: dict[str, Path] = {}
    prog_vels: dict[str, np.ndarray] = {}
    for tk in takes:
        pv = programmed_velocities(tk, level)
        prog_vels[tk.take_id] = pv
        p = workdir / f"{tk.take_id.replace('/', '_')}_L{level}.mid"
        build_programmed_midi(tk, pv).save(str(p))
        prog_paths[tk.take_id] = p
        frames.append(pd.DataFrame({
            "take": tk.take_id,
            "group": tk.groups,
            "pos": tk.grid_pos,
            "slot": tk.grid_ticks,
            "role": take_role_labels(tk),   # fixed human-derived labels, all conditions
            "ord": np.arange(len(tk.notes)),
            "h_off": tk.human_off,
            "h_vel": tk.human_vel.astype(float),
            "h_t": tk.human_t,
            "i_vel": pv.astype(float),
            "i_t": np.array([
                ticks_to_ms_with_map(0, g, tk.tempo_map, tk.ppq) for g in tk.grid_ticks
            ]),
        }))
    df = pd.concat(frames, ignore_index=True)

    groups = [
        g for g in GROUP_ORDER if int((df["group"] == g).sum()) >= MIN_GROUP_HITS
    ]

    results = {
        "human": compute_metrics(
            df, df["h_off"].to_numpy(), df["h_vel"].to_numpy(), df["h_t"].to_numpy(), groups
        ),
        "input": compute_metrics(
            df, np.zeros(len(df)), df["i_vel"].to_numpy(), df["i_t"].to_numpy(), groups
        ),
        "profiles": {},
    }

    for name, prof in loaded_profiles.items():
        per_seed = []
        for seed in seeds:
            offs, vels, ts = [], [], []
            for tj, tk in enumerate(takes):
                out_path = workdir / "out.mid"
                humanise(
                    prog_paths[tk.take_id], out_path, prof,
                    genre="rock", beat_type=tk.beat_type,
                    # Measure at FULL scale regardless of the product's default
                    # intensity: gating compares the engine's reproduction of the
                    # human distributions, and all recorded baselines are at 1.0.
                    intensity=1.0,
                    seed=seed * 1000 + tj,          # distinct RNG stream per take
                )
                o_abs, o_vel = read_output_hits(out_path, tk)
                offs.append(np.array([
                    signed_offset_ms(g, a, tk.tempo_map, tk.ppq)
                    for g, a in zip(tk.grid_ticks, o_abs)
                ]))
                vels.append(o_vel.astype(float))
                ts.append(np.array([
                    ticks_to_ms_with_map(0, a, tk.tempo_map, tk.ppq) for a in o_abs
                ]))
            per_seed.append(compute_metrics(
                df, np.concatenate(offs), np.concatenate(vels), np.concatenate(ts), groups
            ))
        results["profiles"][name] = per_seed
    return results, groups


def aggregate_seeds(per_seed: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    """(mean, 95% CI half-width) across seeds for every (group, metric)."""
    agg: dict[str, dict[str, tuple[float, float]]] = {}
    for g in per_seed[0]:
        agg[g] = {}
        for m in per_seed[0][g]:
            vals = np.array([ps[g][m] for ps in per_seed], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                agg[g][m] = (float("nan"), float("nan"))
            elif len(vals) == 1:
                agg[g][m] = (float(vals[0]), float("nan"))
            else:
                hw = float(t_dist.ppf(0.975, len(vals) - 1) * vals.std(ddof=1) / math.sqrt(len(vals)))
                agg[g][m] = (float(vals.mean()), hw)
    return agg


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: float, hw: float | None = None, prec: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—".rjust(8) + (" " * 8 if hw is not None else "")
    s = f"{v:8.{prec}f}"
    if hw is not None:
        s += f" ±{hw:5.{prec}f}" if not math.isnan(hw) else " " * 8
    return s


def print_report(results: dict, groups: list[str], profile_names: list[str]) -> None:
    agg = {name: aggregate_seeds(results["profiles"][name]) for name in profile_names}
    for g in ["ALL"] + groups:
        click.echo(f"\n── {g} " + "─" * max(0, 100 - len(g)))
        header = f"{'metric':<40}{'human':>9}{'input':>9}"
        for name in profile_names:
            header += f"{name:>17}"
        click.echo(header)
        for key, label, kind in METRIC_ROWS:
            if key not in results["human"][g]:
                continue
            hum = results["human"][g][key]
            inp = results["input"][g][key]
            # distances of human-to-itself are trivial (0 / 1) — blank them out
            hum_s = "     —" if kind == "dist" else _fmt(hum, prec=2).strip().rjust(9)
            line = f"{label:<40}{hum_s:>9}{_fmt(inp).strip():>9}"
            for name in profile_names:
                mean, hw = agg[name][g][key]
                cell = _fmt(mean, hw)
                line += f"{cell.strip():>17}"
            click.echo(line)


def print_sanity(results: dict, profile_names: list[str]) -> None:
    """The explicit checkpoint-1 sanity flags."""
    agg = {name: aggregate_seeds(results["profiles"][name]) for name in profile_names}
    inp = results["input"]["ALL"]
    click.echo("\n" + "=" * 100)
    click.echo("SANITY CHECKS (ALL instruments)")
    for name in profile_names:
        a = agg[name]["ALL"]
        t_closer = a["off_w1"][0] < inp["off_w1"]
        click.echo(
            f"  timing   [{name:>9}]: off_w1 {a['off_w1'][0]:6.2f} vs input {inp['off_w1']:6.2f}"
            f"  → closer to human than quantised input? {'YES' if t_closer else 'NO  ← PROBLEM'}"
        )
    for name in profile_names:
        a = agg[name]["ALL"]
        v_closer = a["vjump_w1"][0] < inp["vjump_w1"]
        click.echo(
            f"  velocity [{name:>9}]: vjump_w1 {a['vjump_w1'][0]:6.2f} vs input {inp['vjump_w1']:6.2f}"
            f"  → closer to human than input? {'YES' if v_closer else 'NO'}"
            "  (old schema expected to lose here — that is the rebuild's motivation)"
        )
    # anti-robotic metrics (addendum Fix 2): TWO-SIDED distance to the human
    # reference — too little spread is robotic, too much is noise. The coarsened
    # input should now legitimately lose these.
    hum = results["human"]["ALL"]
    for m in ("zjump_mass", "wrole_sigma"):
        h = hum[m]
        d_inp = abs(inp[m] - h)
        line = f"  anti-robotic {m:<12}: human {h:6.2f} | input {inp[m]:6.2f} (d={d_inp:5.2f})"
        for name in profile_names:
            v = agg[name]["ALL"][m][0]
            line += f" | {name} {v:6.2f} (d={abs(v - h):5.2f}{'✓' if abs(v - h) < d_inp else ' '})"
        click.echo(line + "   ✓ = closer to human than input")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("gmd_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--candidate", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Rebuilt-schema profile to gate (checkpoint 3).")
@click.option("--gate-profile", type=click.Path(path_type=Path), default=GATE_PROFILE_DEFAULT,
              show_default=True, help="Old-schema train-split profile (built here if missing).")
@click.option("--rebuild-gate", is_flag=True, help="Force rebuilding the gate profile.")
@click.option("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS), show_default=True,
              help="Comma-separated fixed seeds for the profile conditions.")
@click.option("--levels", default="4", show_default=True,
              help="Velocity palette size for the gate run (int).")
@click.option("--sensitivity", is_flag=True,
              help="Also run 2/8/flat input coarseness (gate profile only, context).")
@click.option("--json-out", type=click.Path(path_type=Path), default=REPORT_DEFAULT,
              show_default=True, help="Write the full results as JSON here.")
def main(gmd_dir: Path, candidate: Path | None, gate_profile: Path, rebuild_gate: bool,
         seeds: str, levels: str, sensitivity: bool, json_out: Path) -> None:
    """Run the Part C validation harness against held-out GMD rock takes."""
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    info = pd.read_csv(gmd_dir / "info.csv")
    rock = info[info["style"].str.startswith("rock")]
    train_df = rock[rock["split"] == "train"]
    test_df = rock[rock["split"] == "test"]
    click.echo(f"GMD rock takes: {len(rock)}  (train {len(train_df)}, test {len(test_df)}, "
               f"validation {len(rock) - len(train_df) - len(test_df)} [unused])")

    # -- gate profile: old schema, train split only --------------------------
    if rebuild_gate or not gate_profile.exists():
        click.echo("Building gate profile (old schema, train split only)…")
        raw_hits, skipped = collect_hits(gmd_dir, train_df)
        output, written, skipped_buckets = build_profile_output(raw_hits)
        gate_profile.parent.mkdir(parents=True, exist_ok=True)
        with gate_profile.open("w") as f:
            json.dump(output, f)
        click.echo(f"  gate: {len(raw_hits)} hits from {len(train_df)} takes "
                   f"({skipped} files skipped) → {written} buckets "
                   f"({skipped_buckets} under {30} samples) → {gate_profile}")
    else:
        click.echo(f"Using existing gate profile: {gate_profile}")

    # -- held-out takes -------------------------------------------------------
    takes = []
    dropped = []
    for row in test_df.itertuples():
        tk = load_take(gmd_dir, row)
        if tk is None:
            dropped.append(str(row.id))
        else:
            takes.append(tk)
    total_hits = sum(len(t.notes) for t in takes)
    click.echo(f"Held-out takes evaluated: {len(takes)} ({total_hits} hits); "
               f"dropped {len(dropped)} (missing/non-4/4/<{MIN_EVAL_HITS} hits): {', '.join(dropped)}")

    # -- profiles --------------------------------------------------------------
    loaded = {
        "gate": load_profile(gate_profile),
        "shipped": load_profile(SHIPPED_PROFILE),
    }
    if candidate is not None:
        loaded["candidate"] = load_profile(candidate)
    profile_names = list(loaded)

    gate_level = int(levels)
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        click.echo(f"\nRunning gate level ({gate_level}-level palette), "
                   f"{len(profile_names)} profiles × {len(seed_list)} seeds × {len(takes)} takes…")
        results, groups = evaluate_level(takes, gate_level, loaded, seed_list, workdir)
        print_report(results, groups, profile_names)
        print_sanity(results, profile_names)

        report = {
            "config": {
                "gmd_dir": str(gmd_dir), "seeds": seed_list, "gate_level": gate_level,
                "takes": [t.take_id for t in takes], "dropped": dropped,
                "profiles": {n: str(p) for n, p in
                             [("gate", gate_profile), ("shipped", SHIPPED_PROFILE)]
                             + ([("candidate", candidate)] if candidate else [])},
            },
            "groups": groups,
            "levels": {str(gate_level): {
                "human": results["human"], "input": results["input"],
                "profiles_per_seed": results["profiles"],
                "profiles_agg": {n: aggregate_seeds(results["profiles"][n])
                                 for n in profile_names},
            }},
        }

        if sensitivity:
            click.echo("\n" + "=" * 100)
            click.echo("SENSITIVITY (gate profile only; context, not the gate): "
                       "input coarseness 2 / 8 / flat")
            sens_metrics = ["vel_wpos_sigma", "zjump_mass", "wrole_sigma", "vjump_mean",
                            "vjump_w1", "contour_mae", "spear_orig", "off_w1"]
            for lev in [2, 8, "flat"]:
                r_lev, _ = evaluate_level(
                    takes, lev, {"gate": loaded["gate"]}, seed_list, workdir
                )
                a = aggregate_seeds(r_lev["profiles"]["gate"])["ALL"]
                inp = r_lev["input"]["ALL"]
                hum = r_lev["human"]["ALL"]
                click.echo(f"\n  level={lev} (ALL instruments)")
                click.echo(f"    {'metric':<18}{'human':>9}{'input':>9}{'gate':>17}")
                for m in sens_metrics:
                    mean, hw = a[m]
                    click.echo(f"    {m:<18}{_fmt(hum[m]).strip():>9}"
                               f"{_fmt(inp[m]).strip():>9}{_fmt(mean, hw).strip():>17}")
                report["levels"][str(lev)] = {
                    "human": r_lev["human"], "input": r_lev["input"],
                    "profiles_per_seed": r_lev["profiles"],
                    "profiles_agg": {"gate": aggregate_seeds(r_lev["profiles"]["gate"])},
                }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w") as f:
        json.dump(_jsonable(report), f, indent=1)
    click.echo(f"\nFull results written to {json_out}")


if __name__ == "__main__":
    main()
