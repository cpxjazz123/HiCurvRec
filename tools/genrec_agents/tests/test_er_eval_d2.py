"""The conditional level-2 discrimination's tests (issue #109, section 3.B).

D2 is computed in the SAME frozen read as the enrichment ratio, because the frozen pair
read may happen only once per arm. The tests therefore check two things at once: that the
enrichment ratio's numbers are untouched by the addition, and that D2 says what it says on
cases whose answer is known by construction.

    python3 -m pytest tools/genrec_agents/tests/test_er_eval_d2.py -q

(the interpreter must have numpy; the repository's system python3 does)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# the frozen evaluation needs numpy; the controller's own environment does not have it, so
# the file skips there and is run where it is present
np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = pytest.importorskip("numpy")

from tools.mvg import er_eval  # noqa: E402


def _fixture(tmp_path: Path, *, anchors, positives, negatives, codes):
    """A tiny SID export and a tiny frozen pair archive."""
    sid = np.asarray(codes, dtype=np.int64)
    sid_path = tmp_path / "sids.npy"
    np.save(sid_path, sid)
    archive = tmp_path / "pairs.npz"
    np.savez(
        archive,
        anchors=np.asarray(anchors, dtype=np.int32),
        positives=np.asarray(positives, dtype=np.int32),
        negatives=np.asarray(negatives, dtype=np.int32),
    )
    return sid_path, archive


#: Six items: 0 and 1 share every level, 2 shares the first two levels of 0, and 3, 4, 5
#: live in their own cells.
CODES = [
    [0, 10, 100],
    [0, 10, 100],
    [0, 10, 101],
    [1, 11, 110],
    [1, 11, 111],
    [2, 12, 120],
]


def _evaluate(tmp_path, monkeypatch, **kwargs):
    sid_path, archive = _fixture(tmp_path, **kwargs)
    monkeypatch.setattr(er_eval, "VALID_PAIRS", archive)
    return er_eval.evaluate(sid_path)


def test_the_enrichment_ratio_is_unchanged_by_the_d2(tmp_path, monkeypatch):
    """Adding D2 must not move the ER's numbers."""
    report = _evaluate(
        tmp_path,
        monkeypatch,
        anchors=[0, 3],
        positives=[1, 4],
        negatives=[[2, 5], [5, 2]],
        codes=CODES,
    )
    assert report["ok"] is True
    sid = np.asarray(CODES, dtype=np.int64)
    anchors = np.asarray([0, 3])
    positives = np.asarray([1, 4])
    negatives = np.asarray([[2, 5], [5, 2]])
    expected_positive = float((sid[anchors, 0] == sid[positives, 0]).mean())
    expected_negative = float(
        (sid[anchors, 0][:, None] == sid[negatives, 0]).mean()
    )
    assert report["pos_share"] == pytest.approx(expected_positive)
    assert report["neg_share"] == pytest.approx(expected_negative)
    assert report["l1_er"] == pytest.approx(expected_positive / expected_negative)


def test_a_positive_discrimination_is_reported(tmp_path, monkeypatch):
    """When the positives share L2 and the negatives do not, D2 is positive."""
    report = _evaluate(
        tmp_path,
        monkeypatch,
        anchors=[0],
        positives=[1],
        negatives=[[2]],
        codes=CODES,
    )
    d2 = report["d2"]
    assert d2["ok"] is True
    assert d2["conditional_positive"] == pytest.approx(1.0)
    assert d2["conditional_negative"] == pytest.approx(0.0)
    assert d2["d2"] == pytest.approx(1.0)
    assert d2["numerator_positive"] == 1
    assert d2["denominator_positive"] == 1
    assert d2["numerator_negative"] == 0
    assert d2["denominator_negative"] == 1


def test_a_negative_discrimination_is_reported(tmp_path, monkeypatch):
    """When the negatives share L2 and the positives do not, D2 is negative."""
    report = _evaluate(
        tmp_path,
        monkeypatch,
        anchors=[0],
        positives=[2],
        negatives=[[1]],
        codes=CODES,
    )
    d2 = report["d2"]
    assert d2["conditional_positive"] == pytest.approx(0.0)
    assert d2["conditional_negative"] == pytest.approx(1.0)
    assert d2["d2"] == pytest.approx(-1.0)


def test_an_undefined_conditional_set_is_flagged_and_not_zero(tmp_path, monkeypatch):
    """A conditional set with no pairs is undefined, and must not be reported as zero."""
    codes = [
        [0, 10, 100],
        [1, 11, 110],
        [2, 12, 120],
        [3, 13, 130],
    ]
    report = _evaluate(
        tmp_path,
        monkeypatch,
        anchors=[0],
        positives=[1],
        negatives=[[2]],
        codes=codes,
    )
    d2 = report["d2"]
    assert d2["undefined"] is True
    assert "undefined_zero_denominator" in d2["flags"]
    assert d2["d2"] is None
    assert d2["conditional_positive"] is None


def test_a_small_conditional_set_is_flagged_as_low_support(tmp_path, monkeypatch):
    """A conditional set with few pairs is reported as low support."""
    report = _evaluate(
        tmp_path,
        monkeypatch,
        anchors=[0],
        positives=[1],
        negatives=[[2]],
        codes=CODES,
    )
    d2 = report["d2"]
    assert d2["low_support"] is True
    assert "low_support" in d2["flags"]
    assert d2["undefined"] is False
    assert d2["anchor_cluster_ci95"]["unit"] == "anchor"
    assert d2["anchor_cluster_ci95"]["replicates"] == 1000


def test_the_counts_agree_with_the_probabilities(tmp_path, monkeypatch):
    """The reported numerators and denominators must produce the reported probabilities."""
    report = _evaluate(
        tmp_path,
        monkeypatch,
        anchors=[0, 3],
        positives=[1, 2],
        negatives=[[2, 5], [5, 2]],
        codes=CODES,
    )
    d2 = report["d2"]
    assert d2["conditional_positive"] == pytest.approx(
        d2["numerator_positive"] / d2["denominator_positive"]
    )
    assert d2["conditional_negative"] == pytest.approx(
        d2["numerator_negative"] / d2["denominator_negative"]
    )
    assert d2["d2"] == pytest.approx(
        d2["conditional_positive"] - d2["conditional_negative"]
    )


def test_the_real_control_reads_once_and_reports_its_d2():
    """The archived control: the ER is the archived value and the D2 is defined."""
    sid = REPO_ROOT / "results/stage2_RQ-VAE/r8_control/sids_for_hgrec.npy"
    if not sid.is_file():
        pytest.skip("the archived control's export is not on this machine")
    report = er_eval.evaluate(sid)
    assert report["l1_er"] == pytest.approx(16.334505666275888, abs=1e-9)
    assert report["d2"]["ok"] is True
    assert report["d2"]["d2"] is not None
    assert report["d2"]["denominator_positive"] > 0
