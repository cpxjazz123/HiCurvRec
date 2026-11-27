"""Issue #274: the mutation tests for the pair-ranking loss.

Each mutation is a typical mistake injected into a **copy** of the production module; the copy and
the test files are run in a temporary directory and the catching test is recorded.  A mutation that
only crashes or only produces an import error does **not** count: the run must report a failed
assertion, and the failing test name is the evidence.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

MUTATIONS = [
    ("M1_sign_flip", "d_pos - d_neg + float(margin)", "d_neg - d_pos + float(margin)",
     "the hinge's sign is flipped"),
    ("M2_margin_sign", "d_pos - d_neg + float(margin)", "d_pos - d_neg - float(margin)",
     "the margin enters with the wrong sign"),
    ("M3_wrong_reduction", "return torch.relu(d_pos - d_neg + float(margin)).mean()",
     "return torch.relu(d_pos - d_neg + float(margin)).sum()",
     "the mean over the groups is replaced by a sum"),
    ("M4_single_level_hard_q", "total = selected if total is None else total + selected",
     "total = selected", "the hard q uses only the first level"),
    ("M5_swapped_pairs", "d_pos = torch.linalg.vector_norm(q[pos_b] - q[pos_c], dim=-1)",
     "d_pos = torch.linalg.vector_norm(q[neg_d] - q[neg_e], dim=-1)",
     "the positive distance reads the negative pair"),
    ("M6_holdout_leak", "train_mask = ~self.is_holdout", "train_mask = np.ones_like(self.is_holdout)",
     "the hold-out groups enter the training sampling"),
    ("M7_detached_codebook", "selected = F.embedding(tokens[:, level], codebook)",
     "selected = F.embedding(tokens[:, level], codebook.detach())",
     "the codebook gradient is detached, so the loss cannot train the code geometry"),
]


def _run_mutation(name: str, old: str, new: str) -> dict:
    with tempfile.TemporaryDirectory(dir=TREE) as directory:
        work = Path(directory)
        shutil.copy(TREE / "pair_rank_loss.py", work / "pair_rank_loss.py")
        shutil.copytree(TREE / "tests", work / "tests")
        source = (work / "pair_rank_loss.py").read_text()
        if old not in source:
            raise AssertionError(f"the mutation {name} does not match the production source")
        (work / "pair_rank_loss.py").write_text(source.replace(old, new, 1))
        completed = subprocess.run(
            [PYTHON, "-m", "pytest", "tests/test_known_answers.py", "tests/test_pair_rank.py",
             "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=work, capture_output=True, text=True, timeout=900)
        output = completed.stdout + completed.stderr
        failed = [line.split("::")[-1].split(" ")[0]
                  for line in output.splitlines() if line.startswith("FAILED ")]
        errors = [line for line in output.splitlines() if line.startswith("ERROR ")]
        return {"mutation": name, "returncode": completed.returncode, "failed_tests": failed,
                "errors": errors, "caught": bool(failed) and not errors}


def test_every_mutation_is_caught_by_an_assertion():
    results = []
    for name, old, new, _ in MUTATIONS:
        results.append(_run_mutation(name, old, new))
    report = {entry["mutation"]: entry for entry in results}
    (TREE / "MUTATIONS.json").write_text(
        __import__("json").dumps({"schema": "genrec.issue274.mutations/1", "results": results},
                                indent=2, sort_keys=True) + "\n")
    for entry in results:
        assert entry["caught"], f"{entry['mutation']} was not caught by an assertion: {entry}"
    assert len(results) >= 5
