"""Issue #220 CGV log: the six checks, their evidence files and the replay commands."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("A", "B", "C", "D", "E")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"


def main() -> None:
    out = HERE / "cgv"
    out.mkdir(exist_ok=True)
    tests = subprocess.run([PYTHON, "-m", "pytest", str(HERE / "template" / "tests"), "-q"],
                           capture_output=True, text=True, cwd=str(HERE.parents[2]))
    summary = tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else ""
    entry = {
        "issue": 220,
        "items": {
            "1_labels_forward": "test_trq.py::test_labels_come_from_the_graph_not_from_the_input, "
                                "::test_head_is_permutation_invariant_and_equivariant, "
                                "::test_cross_entropies_are_finite_and_positive, "
                                "test_trq_data.py::test_pools_are_strictly_induced",
            "2_gradient_path": "test_trq.py::test_arm_b_gradients_reach_encoder_not_codebook, "
                               "::test_arm_c_ste_codebook_gradient_is_zero, "
                               "::test_arm_d_gradients_reach_encoder_and_codebook, "
                               "::test_arm_e_frozen_head_stays_frozen",
            "3_direction": "test_trq.py::test_one_step_reduces_the_joint_loss, "
                           "::test_ste_onehot_matches_the_hard_onehot_in_forward",
            "4_ddp": "the MCP smoke for each arm (launch_mode=ddp, applied grad norms > 0) "
                     "plus the four-rank 3000-epoch runs",
            "5_entry_and_export": "MCP one-epoch smoke per arm + export_determinism.py",
            "6_data_and_health": "test_trq_data.py (pool strictness, split disjointness, graph SHA, "
                                 "role masks, health stops)",
        },
        "pytest": {"command": f"{PYTHON} -m pytest {HERE / 'template' / 'tests'} -q",
                   "exit_code": tests.returncode, "summary": summary},
        "determinism": {arm: json.loads((HERE / f"determinism_{arm}.json").read_text())
                        for arm in ARMS if (HERE / f"determinism_{arm}.json").is_file()},
        "replay": [
            f"{PYTHON} results/research/trq/pretrain_head.py",
            f"{PYTHON} -m pytest results/research/trq/template/tests -q",
            "python3 results/research/trq/make_arm.py <A|B|C|D|E>",
            "stage2_start(source_dir=<tree>) then stage2_status(source_dir=<tree>)",
            f"{PYTHON} results/research/trq/evaluate.py",
            f"{PYTHON} results/research/trq/export_determinism.py <arm>",
            "python3 results/research/trq/manifest.py",
            "python3 results/research/trq/report.py",
        ],
    }
    (out / "cgv_log.json").write_text(json.dumps(entry, indent=1))
    print(summary, "exit", tests.returncode)


if __name__ == "__main__":
    main()
