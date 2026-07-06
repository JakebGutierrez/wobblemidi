"""Golden-vector byte locks (porting contract).

Every test run replays each checked-in vector through the current engine and
byte-compares against tests/golden/outputs/. A failure here means engine behaviour
changed: either fix the regression, or — for an INTENTIONAL behaviour change —
regenerate the vectors deliberately (scripts/make_golden.py --force) in the same
commit and say so in its message. See wobblemidi_determinism.md.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from verify_golden import (   # noqa: E402
    GOLDEN_DIR,
    check_profile,
    check_seed_pairs,
    check_vector,
    load_bundled_profile,
    load_manifest,
)

_MANIFEST = load_manifest()


@pytest.fixture(scope="module")
def profile_and_sha():
    return load_bundled_profile()


def test_bundled_profile_matches_manifest(profile_and_sha):
    _, sha = profile_and_sha
    failure = check_profile(_MANIFEST, sha)
    assert failure is None, failure


@pytest.mark.parametrize(
    "entry", _MANIFEST["vectors"], ids=[v["id"] for v in _MANIFEST["vectors"]]
)
def test_vector_byte_identical(entry, profile_and_sha, tmp_path):
    profile, _ = profile_and_sha
    failure = check_vector(entry, profile, GOLDEN_DIR, tmp_path)
    assert failure is None, failure


def test_seed_sensitivity_pairs_differ():
    failures = check_seed_pairs(_MANIFEST, GOLDEN_DIR)
    assert not failures, "\n".join(failures)
