"""Mutation testing for issue #243 (CONTRACT §6, C5).

Each mutation is a realistic defect injected into the sources, run against the real test
suite in an isolated copy.  A mutation that survives means the tests have no detection
power for that failure mode, which is itself a finding.

Mutations cover the three classes the contract asks for:
numeric (coefficient, clamping), structural (dropped edges, wrong reduction), and
routing (gradient cut, wrong metric).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
WORK = HERE / "_mutation"
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

#: (name, class, file, old, new, test that must fail)
MUTATIONS = [
    ("dist-coefficient-twice", "numeric", "manifold.py",
     "return 2.0 * tan_k_inv(moved.norm(dim=-1), k)",
     "return 4.0 * tan_k_inv(moved.norm(dim=-1), k)",
     "tests/test_reference.py"),
    ("drop-out-edges", "structural", "kappa_gcn.py",
     "        if out_edges.numel():",
     "        if False:",
     "tests/test_reference.py"),
    ("proju-no-clamp", "numeric", "manifold.py",
     "    scale = torch.where(norm > maxnorm, maxnorm / norm.clamp(min=EPS), torch.ones_like(norm))\n    return x * scale",
     "    return x",
     "tests/test_reference.py"),
    ("expmap0-detached", "routing", "manifold.py",
     "    return proju(scale * x, k)",
     "    return proju(scale * x, k).detach()",
     "tests/test_reference.py"),
    ("spearman-as-pearson", "routing", "reference.py",
     "    rx = ranks(x) - ranks(x).mean()\n    ry = ranks(y) - ranks(y).mean()",
     "    rx = x - x.mean()\n    ry = y - y.mean()",
     "tests/test_known_answers.py"),
    ("mobius-add-sign", "numeric", "manifold.py",
     "    numerator = (1.0 - 2.0 * k * xy - k * y2) * x + (1.0 + k * x2) * y",
     "    numerator = (1.0 - 2.0 * k * xy - k * y2) * x - (1.0 + k * x2) * y",
     "tests/test_reference.py"),
]


def copy_tree(destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(HERE, destination,
                    ignore=shutil.ignore_patterns("_mutation", "__pycache__", "*.pyc"))
    return destination


def run_tests(tree: Path, test_file: str) -> subprocess.CompletedProcess:
    return subprocess.run([PYTHON, test_file], cwd=tree, capture_output=True, text=True, timeout=600)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    survivors = []
    for name, kind, source, old, new, test_file in MUTATIONS:
        tree = copy_tree(WORK / name)
        target = tree / source
        text = target.read_text(encoding="utf-8")
        if old not in text:
            rows.append((name, kind, "ANCHOR-MISSING", "n/a"))
            survivors.append(name)
            continue
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        baseline = run_tests(tree, test_file)
        status = "CAUGHT" if baseline.returncode != 0 else "SURVIVED"
        first_fail = ""
        for line in baseline.stdout.splitlines():
            if line.startswith("FAIL"):
                first_fail = line.strip()[:120]
                break
        rows.append((name, kind, status, first_fail))
        if status != "CAUGHT":
            survivors.append(name)
        shutil.rmtree(tree)

    lines = ["# Issue #243 变异测试", "",
             f"共 {len(MUTATIONS)} 个变异；被抓到 {len(MUTATIONS) - len(survivors)} 个。", "",
             "| 变异 | 类别 | 结果 | 抓到的测试 |", "|---|---|---|---|"]
    for name, kind, status, evidence in rows:
        lines.append(f"| `{name}` | {kind} | **{status}** | {evidence} |")
    if survivors:
        lines.append("")
        lines.append("存活变异（测试无检测力）: " + ", ".join(survivors))
    (HERE / "MUTATIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if not survivors else 1


if __name__ == "__main__":
    raise SystemExit(main())
