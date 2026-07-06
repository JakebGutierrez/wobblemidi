"""Verify the golden vectors: replay every manifest entry and byte-compare the output.

Read-only — the check half of the porting contract (see wobblemidi_determinism.md and
wobblemidi_porting_contract.md). Regeneration is deliberately NOT offered here; it lives
behind `scripts/make_golden.py --force` and must accompany an intentional engine/profile
behaviour change.

Checks, per vector:
  1. the checked-in input fixture still matches its recorded sha256 (edits require regen);
  2. the checked-in output still matches its recorded sha256 (manifest/output drift);
  3. replaying the vector through the CURRENT engine reproduces the output byte-for-byte
     (on mismatch: environment drift is reported first, then the first event divergence);
plus, once per run: the bundled profile hash matches the manifest (a profile rebuild
invalidates the vectors), and the seed-sensitivity pairs produce different bytes.

Usage:
    python scripts/verify_golden.py [--verbose]     # exit 0 = contract holds

pytest runs the same checks via tests/test_golden_vectors.py.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
from importlib.metadata import version as pkg_version
from importlib.resources import as_file, files
from pathlib import Path

import click
import mido

REPO_ROOT = Path(__file__).parent.parent
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

from wobblemidi.humanise import humanise, load_profile   # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(golden_dir: Path = GOLDEN_DIR) -> dict:
    return json.loads((golden_dir / "manifest.json").read_text())


def load_bundled_profile():
    """Return (LoadedProfile, sha256-of-json) for the bundled rock profile."""
    with as_file(files("wobblemidi.profiles").joinpath("rock.json")) as p:
        return load_profile(p), _sha256(Path(p))


def environment_drift(manifest: dict) -> list[str]:
    """Recorded-vs-current environment differences (informational; the first thing
    to suspect on a byte mismatch)."""
    meta = manifest["_meta"]
    current = {
        "python": platform.python_version(),
        "numpy": pkg_version("numpy"),
        "scipy": pkg_version("scipy"),
        "mido": pkg_version("mido"),
        "platform": platform.platform(),
    }
    return [
        f"{key}: recorded {meta.get(key)!r} vs current {value!r}"
        for key, value in current.items()
        if meta.get(key) != value
    ]


def check_profile(manifest: dict, profile_sha: str) -> str | None:
    recorded = manifest["_meta"]["profile_sha256"]
    if recorded != profile_sha:
        return (
            f"bundled profile sha256 {profile_sha[:12]}… does not match the manifest's "
            f"{recorded[:12]}… — the vectors were generated against a different "
            "wobblemidi/profiles/rock.json. A profile rebuild requires deliberately "
            "regenerating the golden vectors (scripts/make_golden.py --force) in the "
            "same commit."
        )
    return None


def replay_vector(entry: dict, profile, golden_dir: Path, out_path: Path) -> None:
    """Run one manifest entry through the current engine (or the CLI) into out_path."""
    input_path = golden_dir / entry["input"]
    if "cli_args" in entry:
        from click.testing import CliRunner
        from wobblemidi.cli import main as cli_main
        result = CliRunner().invoke(
            cli_main, [str(input_path), str(out_path), *entry["cli_args"]])
        if result.exit_code != 0:
            raise RuntimeError(
                f"{entry['id']}: CLI exited {result.exit_code}: {result.output}")
    else:
        humanise(input_path, out_path, profile, **entry["params"])


def first_divergence(expected_path: Path, got_path: Path, context: int = 3) -> str:
    """Human-readable first event-level difference between two MIDI files."""
    exp = mido.MidiFile(str(expected_path))
    got = mido.MidiFile(str(got_path))
    if exp.type != got.type or exp.ticks_per_beat != got.ticks_per_beat:
        return (f"header differs: type {exp.type}/{got.type}, "
                f"ppq {exp.ticks_per_beat}/{got.ticks_per_beat}")
    if len(exp.tracks) != len(got.tracks):
        return f"track count differs: expected {len(exp.tracks)}, got {len(got.tracks)}"
    for ti, (te, tg) in enumerate(zip(exp.tracks, got.tracks)):
        for mi, (me, mg) in enumerate(zip(te, tg)):
            if me != mg:
                lines = [f"track {ti}, message {mi}:"]
                lo = max(0, mi - context)
                for j in range(lo, min(mi + context + 1, len(te), len(tg))):
                    marker = ">>" if j == mi else "  "
                    lines.append(f"  {marker} expected[{j}]: {te[j]}")
                    lines.append(f"  {marker}      got[{j}]: {tg[j]}")
                return "\n".join(lines)
        if len(te) != len(tg):
            return (f"track {ti} length differs: expected {len(te)} messages, "
                    f"got {len(tg)}")
    return "files decode identically but bytes differ (encoding-level difference)"


def check_vector(entry: dict, profile, golden_dir: Path, tmp_dir: Path,
                 drift: list[str] | None = None) -> str | None:
    """Return a failure description for one vector, or None if it verifies."""
    vector_id = entry["id"]
    input_path = golden_dir / entry["input"]
    output_path = golden_dir / entry["output"]

    if not input_path.exists():
        return f"{vector_id}: missing input fixture {entry['input']}"
    if _sha256(input_path) != entry["sha256_input"]:
        return (f"{vector_id}: input fixture {entry['input']} does not match its recorded "
                "sha256 — fixtures are frozen; changing one requires regenerating the "
                "vectors deliberately (scripts/make_golden.py --force).")
    if not output_path.exists():
        return f"{vector_id}: missing stored output {entry['output']}"
    if _sha256(output_path) != entry["sha256_output"]:
        return (f"{vector_id}: stored output {entry['output']} does not match the "
                "manifest's sha256 — output/manifest drift; regenerate deliberately.")

    replayed = tmp_dir / f"{vector_id}.mid"
    replay_vector(entry, profile, golden_dir, replayed)
    if replayed.read_bytes() == output_path.read_bytes():
        return None

    lines = [f"{vector_id}: replayed output differs from the stored golden vector."]
    if drift:
        lines.append("Environment drift (suspect first):")
        lines += [f"  - {d}" for d in drift]
    else:
        lines.append("Environment matches the recorded one — this is an ENGINE "
                     "behaviour change.")
    lines.append(first_divergence(output_path, replayed))
    return "\n".join(lines)


def check_seed_pairs(manifest: dict, golden_dir: Path) -> list[str]:
    failures = []
    for a, b in manifest["_meta"].get("seed_sensitivity_pairs", []):
        by_id = {e["id"]: e for e in manifest["vectors"]}
        pa = golden_dir / by_id[a]["output"]
        pb = golden_dir / by_id[b]["output"]
        if pa.read_bytes() == pb.read_bytes():
            failures.append(f"seed-sensitivity pair ({a}, {b}) produced IDENTICAL bytes "
                            "— the seed is not reaching the engine.")
    return failures


def run(verbose: bool = False) -> int:
    manifest = load_manifest()
    profile, profile_sha = load_bundled_profile()
    drift = environment_drift(manifest)
    failures: list[str] = []

    profile_failure = check_profile(manifest, profile_sha)
    if profile_failure:
        # Every replay would fail against a different profile; stop early and clearly.
        print(f"FAIL: {profile_failure}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for entry in manifest["vectors"]:
            failure = check_vector(entry, profile, GOLDEN_DIR, tmp_dir, drift=drift)
            if failure:
                failures.append(failure)
                print(f"FAIL {entry['id']}")
            elif verbose:
                print(f"ok   {entry['id']}")

    failures += check_seed_pairs(manifest, GOLDEN_DIR)

    n = len(manifest["vectors"])
    if failures:
        print(f"\n{len(failures)} failure(s) / {n} vectors:\n")
        for f in failures:
            print(f + "\n")
        return 1
    drift_note = f" (environment drift: {'; '.join(drift)})" if drift else ""
    print(f"golden vectors OK: {n}/{n} byte-identical, "
          f"{len(manifest['_meta'].get('seed_sensitivity_pairs', []))} seed pair(s) "
          f"distinct{drift_note}")
    return 0


@click.command()
@click.option("--verbose", is_flag=True, help="Print every vector, not just failures.")
def main(verbose: bool) -> None:
    """Verify the golden-vector contract against the current engine."""
    sys.exit(run(verbose=verbose))


if __name__ == "__main__":
    main()
