"""Mutation testing for the v2 infrastructure (issue #246).

Each mutation injects a realistic defect, runs the real suite in an isolated copy, and
requires the **designated assertion** to fail.  A crash or an import error does not count.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[1]
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
WORK = REPO / "results" / "research" / "infra246" / "_mutation"

MUTATIONS = [
    ("held-out-uses-the-v1-weighted-formula", "numeric", "isolation.py",
     "    support_v = (visible > 0).astype(np.int8).tocsr()\n"
     "    support_e = (evaluation > 0).astype(np.int8).tocsr()\n"
     "    difference = support_e - support_e.multiply(support_v).astype(np.int8)",
     "    symmetric = (evaluation - evaluation.multiply(visible)) > 0\n"
     "    return symmetric.astype(np.int8).tocsr()\n\n"
     "def _unused(visible, evaluation):\n"
     "    support_v = (visible > 0).astype(np.int8).tocsr()\n"
     "    support_e = (evaluation > 0).astype(np.int8).tocsr()\n"
     "    difference = support_e - support_e.multiply(support_v).astype(np.int8)",
     "tests/test_v2_graph.py"),
    ("ego-role-from-a-truncated-sample", "structural", "topology.py",
     "    out_side = set(int(x) for x in _neighbours(out_csr, root)) - {int(root)}",
     "    out_side = set(int(x) for x in _neighbours(out_csr, root)[:8].tolist()) - {int(root)}",
     "tests/test_v2_ego.py"),
    ("ego-hop-equals-traversal-depth", "structural", "topology.py",
     "        true_hop = np.array([0 if int(node) == root else 1 if int(node) in first_hop else 2\n"
     "                             for node in nodes.tolist()], dtype=np.int64)",
     "        true_hop = np.array([depth[int(node)] for node in nodes.tolist()], dtype=np.int64)",
     "tests/test_v2_ego.py"),
    ("fork-back-to-universal-independence", "structural", "graphlet.py",
     "    successors = sorted(_csr_set(out_csr, root))\n"
     "    for index, left in enumerate(successors):\n"
     "        for right in successors[index + 1:]:\n"
     "            if right not in _csr_set(out_csr, left) and left not in _csr_set(out_csr, right):\n"
     "                return True\n"
     "    return False",
     "    successors = sorted(_csr_set(out_csr, root))\n"
     "    for index, left in enumerate(successors):\n"
     "        for right in successors[index + 1:]:\n"
     "            if right in _csr_set(out_csr, left) or left in _csr_set(out_csr, right):\n"
     "                return False\n"
     "    return len(successors) >= 2",
     "tests/test_v2_graphlet.py"),
    ("canonical-code-not-minimised", "numeric", "graphlet.py",
     "    return int(best_code)",
     "    return int(code)",
     "tests/test_v2_graphlet.py"),
    ("subgraph-summary-flips-the-pass-rule", "routing", "metrics.py",
     "        passed = per_subgraph <= threshold",
     "        passed = per_subgraph >= threshold",
     "tests/test_v2_metrics.py"),
    ("subgraph-summary-worst-becomes-the-best", "numeric", "metrics.py",
     "        worst = np.array([values.max() for values in valid], dtype=np.float64)",
     "        worst = np.array([values.min() for values in valid], dtype=np.float64)",
     "tests/test_v2_metrics.py"),
    ("prefix-at-k-drops-the-length-check", "structural", "metrics.py",
     "        if len(left) < k or len(right) < k:\n            continue",
     "        if False:\n            continue",
     "tests/test_v2_metrics.py"),
    ("ppr-drops-the-dangling-correction", "numeric", "ppr.py",
     "        beta = alpha * mass / np.maximum(1.0 - (1.0 - alpha) * mass, 1e-300)",
     "        beta = 0.0 * mass / np.maximum(1.0 - (1.0 - alpha) * mass, 1e-300)",
     "tests/test_v2_ppr.py"),
    ("join-accepts-extra-pairs", "structural", "assets.py",
     "    if extra:\n        raise ValueError(f\"{len(extra)} model pairs absent from the bank, e.g. {extra[:3]}\")",
     "    if False:\n        raise ValueError(f\"{len(extra)} model pairs absent from the bank, e.g. {extra[:3]}\")",
     "tests/test_v2_assets.py"),
    ("reference-flat-limit-reverts-to-plain-norm", "numeric", "reference.py",
     "        return 2.0 * np.linalg.norm(x - y, axis=-1)",
     "        return np.linalg.norm(x - y, axis=-1)",
     "tests/test_known_answers.py"),
    ("manifold-flat-limit-reverts-to-plain-norm", "numeric", "manifold.py",
     "        return 2.0 * (x - y).norm(dim=-1)",
     "        return (x - y).norm(dim=-1)",
     "tests/test_v2_geometry.py"),
    ("sphere-exponential-uses-the-half-angle", "numeric", "manifold.py",
     "        mapped = radius * torch.tan(norm / radius)",
     "        mapped = radius * torch.tan(norm / (2.0 * radius))",
     "tests/test_v2_geometry.py"),
    ("curvature-transport-ignores-the-marginals", "numeric", "curvature.py",
     "    b_eq = np.concatenate([np.full(n, 1.0 / n), np.full(m, 1.0 / m)])",
     "    b_eq = np.concatenate([np.full(n, 1.0), np.full(m, 1.0 / m)])",
     "tests/test_v2_geometry.py"),
]


def prepare(name: str, source: str, old: str, new: str) -> tuple[Path, bool]:
    """Build a fake repository root whose tools.graph_infra is a mutated copy.

    The suite's own ``sys.path`` insert points at ``parents[3]`` of the test file, so placing
    the copy at ``<root>/tools/graph_infra`` makes the test import the mutated package.
    """
    root = WORK / name
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(HERE, root / "tools" / "graph_infra",
                    ignore=shutil.ignore_patterns("_mutation", "__pycache__"))
    target = root / "tools" / "graph_infra" / source
    text = target.read_text()
    if old not in text:
        return root, False
    target.write_text(text.replace(old, new, 1))
    return root, True


def run_suite(root: Path, test_name: str) -> subprocess.CompletedProcess:
    return subprocess.run([PYTHON, f"tools/graph_infra/tests/{test_name}"], cwd=root,
                          capture_output=True, text=True, timeout=900,
                          env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/bin",
                               "HOME": str(Path.home())})


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    rows, survivors = [], []

    # baseline: the unmutated copy must pass, otherwise "caught" means nothing
    baseline_root = WORK / "_baseline"
    if baseline_root.exists():
        shutil.rmtree(baseline_root)
    shutil.copytree(HERE, baseline_root / "tools" / "graph_infra",
                    ignore=shutil.ignore_patterns("_mutation", "__pycache__"))
    baseline_ok = {}
    for test_name in sorted({m[5].split("/")[-1] for m in MUTATIONS}):
        result = run_suite(baseline_root, test_name)
        baseline_ok[test_name] = result.returncode == 0
        if result.returncode != 0:
            print(f"BASELINE-FAIL {test_name}: {result.stdout[-200:]}")
    shutil.rmtree(baseline_root)

    for name, kind, source, old, new, test_file in MUTATIONS:
        test_name = test_file.split("/")[-1]
        root, applied = prepare(name, source, old, new)
        if not applied:
            rows.append((name, kind, "ANCHOR-MISSING", ""))
            survivors.append(name)
            continue
        result = run_suite(root, test_name)
        fail_line = ""
        for line in result.stdout.splitlines():
            if line.startswith("FAIL"):
                fail_line = line.strip()[:120]
                break
        if not baseline_ok.get(test_name, False):
            status = "BASELINE-BROKEN"
        elif result.returncode == 1 and fail_line:
            status = "CAUGHT"
        else:
            status = "SURVIVED"
        rows.append((name, kind, status, fail_line))
        if status != "CAUGHT":
            survivors.append(name)
        shutil.rmtree(root)

    lines = ["# Issue #246 变异测试 (v2)", "",
             f"共 {len(MUTATIONS)} 个变异；被抓到 {len(MUTATIONS) - len(survivors)} 个。",
             "判定 = 未变异副本先通过(returncode 0) 且 变异副本 returncode 1 且输出含 `FAIL` 行；崩溃不计。", "",
             "| 变异 | 类别 | 结果 | 抓到的断言 |", "|---|---|---|---|"]
    for name, kind, status, evidence in rows:
        lines.append(f"| `{name}` | {kind} | **{status}** | {evidence} |")
    if survivors:
        lines.append("")
        lines.append("未捕获: " + ", ".join(survivors))
    output = REPO / "results" / "research" / "infra246" / "MUTATIONS_V2.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    return 0 if not survivors else 1


if __name__ == "__main__":
    raise SystemExit(main())
