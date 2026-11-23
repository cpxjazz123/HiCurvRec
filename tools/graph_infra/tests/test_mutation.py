"""Mutation testing for the shared infrastructure (issue #245 §7).

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
WORK = REPO / "results" / "research" / "infra245" / "_mutation"

MUTATIONS = [
    ("dist-coefficient-twice", "numeric", "manifold.py",
     "    return 2.0 * tan_k_inv(moved.norm(dim=-1), k)",
     "    return 4.0 * tan_k_inv(moved.norm(dim=-1), k)",
     "tests/test_reference.py"),
    ("sphere-distance-halved", "numeric", "manifold.py",
     "        return radius * torch.acos((numerator / denominator).clamp(-1.0, 1.0))",
     "        return 0.5 * radius * torch.acos((numerator / denominator).clamp(-1.0, 1.0))",
     "tests/test_reference.py"),
    ("sphere-logmap-drops-radius", "numeric", "manifold.py",
     "        mapped = 2.0 * radius * torch.atan(norm / radius)  # theta = 2R*arctan(r/R)",
     "        mapped = 2.0 * torch.atan(norm / radius)",
     "tests/test_reference.py"),
    ("sphere-proju-hyperbolic-clamp", "structural", "manifold.py",
     "    if k > 0:\n        guard = 1e6 / math.sqrt(k)\n        scale = torch.where(norm > guard, guard / norm.clamp(min=EPS), torch.ones_like(norm))\n        return x * scale",
     "    if k > 0:\n        maxnorm = (1.0 - eps) / math.sqrt(abs(k))\n        scale = torch.where(norm > maxnorm, maxnorm / norm.clamp(min=EPS), torch.ones_like(norm))\n        return x * scale",
     "tests/test_reference.py"),
    ("mobius-add-sign", "numeric", "manifold.py",
     "    numerator = (1.0 - 2.0 * k * xy - k * y2) * x + (1.0 + k * x2) * y",
     "    numerator = (1.0 - 2.0 * k * xy - k * y2) * x - (1.0 + k * x2) * y",
     "tests/test_reference.py"),
    ("expmap0-detached", "routing", "manifold.py",
     "    return proju(scale * x, k)",
     "    return proju(scale * x, k).detach()",
     "tests/test_reference.py"),
    ("drop-out-edges", "structural", "kappa_gcn.py",
     "        if out_edges.numel():", "        if False:",
     "tests/test_reference.py"),
    ("ego-drops-induced-edges", "structural", "topology.py",
     "            if target in local:\n                edges.append((local[int(node)], local[target]))",
     "            if target in local and int(node) < target:\n                edges.append((local[int(node)], local[target]))",
     "tests/test_topology.py"),
    ("pairs-bucket-by-source", "structural", "pairs.py",
     "            buckets[target].append(node)",
     "            buckets[node].append(target)",
     "tests/test_pairs.py"),
    ("metrics-spearman-as-pearson", "routing", "metrics.py",
     "    rx = average_ranks(x)\n    ry = average_ranks(y)",
     "    rx = x - x.mean()\n    ry = y - y.mean()",
     "tests/test_metrics.py"),
    ("bfs-depth-ignored", "numeric", "graph_distance.py",
     "    if depth is not None:\n        distance[distance > depth] = np.inf\n    return distance",
     "    return distance",
     "tests/test_graph_distance.py"),
    ("join-accepts-missing", "structural", "assets.py",
     "    if missing:\n        raise ValueError(f\"{len(missing)} bank pairs missing from the model output, e.g. {missing[:3]}\")",
     "    if False:\n        raise ValueError(f\"{len(missing)} bank pairs missing from the model output, e.g. {missing[:3]}\")",
     "tests/test_assets.py"),
    ("chain-drops-induced-check", "structural", "graphlet.py",
     "            if second != root and not _has_edge(undirected, root, second):",
     "            if second != root:",
     "tests/test_graphlet.py"),
    ("fork-accepts-adjacent-successors", "structural", "graphlet.py",
     "    def _independent(nodes: set[int]) -> bool:\n        listed = sorted(nodes)",
     "    def _independent(nodes: set[int]) -> bool:\n        return True\n        listed = sorted(nodes)",
     "tests/test_graphlet.py"),
    ("ppr-dangling-mass-dropped", "numeric", "graph_distance.py",
     "        if dangling_mass > 0.0:\n            updated = updated + (1.0 - restart) * dangling_mass * teleport",
     "        if False:\n            updated = updated + (1.0 - restart) * dangling_mass * teleport",
     "tests/test_graph_distance.py"),
    ("two-hop-ignores-reverse-edge", "structural", "pairs.py",
     "                if target == node or _has(out_csr, node, target) or _has(out_csr, target, node):",
     "                if target == node or _has(out_csr, node, target):",
     "tests/test_pairs.py"),
    ("undirected-exclusion-ignored", "structural", "pairs.py",
     "        if blocked[left, right] > 0 or blocked[right, left] > 0:",
     "        if False:",
     "tests/test_pairs.py"),
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

    lines = ["# Issue #245 变异测试", "",
             f"共 {len(MUTATIONS)} 个变异；被抓到 {len(MUTATIONS) - len(survivors)} 个。",
             "判定 = 未变异副本先通过(returncode 0) 且 变异副本 returncode 1 且输出含 `FAIL` 行；崩溃不计。", "",
             "| 变异 | 类别 | 结果 | 抓到的断言 |", "|---|---|---|---|"]
    for name, kind, status, evidence in rows:
        lines.append(f"| `{name}` | {kind} | **{status}** | {evidence} |")
    if survivors:
        lines.append("")
        lines.append("未捕获: " + ", ".join(survivors))
    output = REPO / "results" / "research" / "infra245" / "MUTATIONS.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    return 0 if not survivors else 1


if __name__ == "__main__":
    raise SystemExit(main())
