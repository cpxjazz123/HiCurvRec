"""Issue #287 round 6: the mutation tests for the hard-negative ranking loss.

Each mutation is injected into a **copy** of the production module; the copy and the tests run in a
temporary directory and the catching test is recorded.  A crash or an import error does not count:
the run must report a failed assertion.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

MUTATIONS = [
    ("M1_sign_flip", "d_pos - d_neg + float(margin)", "d_neg - d_pos + float(margin)",
     "the hinge's sign is flipped"),
    ("M2_margin_sign", "d_pos - d_neg + float(margin)", "d_pos - d_neg - float(margin)",
     "the margin enters with the wrong sign"),
    ("M3_wrong_reduction", "return (hinge * weight).sum() / weight.sum()",
     "return (hinge * weight).sum()",
     "the weighted mean over the groups is replaced by a weighted sum"),
    ("M4_swapped_pairs", "d_pos = torch.linalg.vector_norm(z[pos] - z[anchor], dim=-1)",
     "d_pos = torch.linalg.vector_norm(z[neg] - z[anchor], dim=-1)",
     "the positive distance reads the negative item"),
    ("M5_detached_representation", "d_pos = torch.linalg.vector_norm(z[pos] - z[anchor], dim=-1)",
     "d_pos = torch.linalg.vector_norm(z[pos].detach() - z[anchor], dim=-1)",
     "the positive is detached, so the loss cannot train the encoder"),
    ("M6_holdout_leak", "train_mask = ~self.is_holdout", "train_mask = np.ones_like(self.is_holdout)",
     "the hold-out groups enter the training sampling"),
    ("M7_mining_keeps_the_neighbours", "neighbours = set(index.neighbours_of(anchor).tolist())",
     "neighbours = set()",
     "the miner stops excluding the anchor's direct neighbours"),
    ("M8_no_margin_validation", "    if not (margin > 0):\n        raise ValueError(\"the margin must be positive\")\n",
     "", "the loss accepts a non-positive margin"),
    ("M9_weight_ignored", "    if weight is None:\n        return hinge.mean()\n",
     "    if True:\n        return hinge.mean()\n", "the bucket-size weighting is ignored"),
]


def _run_mutation(name: str, old: str, new: str) -> dict:
    with tempfile.TemporaryDirectory(dir=TREE) as directory:
        work = Path(directory)
        shutil.copy(TREE / "hard_negative_loss.py", work / "hard_negative_loss.py")
        shutil.copytree(TREE / "tests", work / "tests")
        source = (work / "hard_negative_loss.py").read_text()
        if old not in source:
            raise AssertionError(f"the mutation {name} does not match the production source")
        (work / "hard_negative_loss.py").write_text(source.replace(old, new, 1))
        completed = subprocess.run(
            [PYTHON, "-m", "pytest", "tests/test_hard_negative.py", "-q", "--no-header",
             "-p", "no:cacheprovider"],
            cwd=work, capture_output=True, text=True, timeout=900)
        output = completed.stdout + completed.stderr
        failed = [line.split("::")[-1].split(" ")[0]
                  for line in output.splitlines() if line.startswith("FAILED ")]
        errors = [line.split("::")[-1].split(" ")[0]
                  for line in output.splitlines() if line.startswith("ERROR ")]
        return {"mutation": name, "returncode": completed.returncode, "failed_tests": failed,
                "errors": errors, "caught": bool(failed) and not errors}


def test_every_mutation_is_caught_by_an_assertion():
    results = [_run_mutation(name, old, new) for name, old, new, _ in MUTATIONS]
    (TREE / "MUTATIONS.json").write_text(json.dumps(
        {"schema": "genrec.issue287.round7.mutations/1", "results": results}, indent=2, sort_keys=True) + "\n")
    for entry in results:
        assert entry["caught"], f"{entry['mutation']} was not caught by an assertion: {entry}"
    assert len(results) >= 5
