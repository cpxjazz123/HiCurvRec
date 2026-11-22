"""One executable matrix: which collapse mode is refused by which check.

The individual pieces are unit-tested; this runs them against one tree each, on the
frozen baseline, and prints the matrix a reviewer wants. Exit 0 only when every case gives
the expected verdict -- so it doubles as a regression gate for the guard itself.

    python tools/research/stage2/anticollapse/acceptance.py [--tree <copy dir>]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage2.anticollapse import (CodebookCollapse, ReadOnlyBaseline, UnknownTree,  # noqa: E402
                                 apply_floor, verify_tree)
from stage2.anticollapse.floor import (constrained_assignment, verify_codebook,  # noqa: E402
                                       verify_margins, verify_separation)
from stage2.anticollapse.patch_tree import floor_constant, unguarded_sites  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"


def _fresh(root: Path, name: str) -> Path:
    tree = root / name
    if tree.exists():
        shutil.rmtree(tree)
    tree.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASELINE, tree, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "logs"))
    return tree


def case_classic_collapse() -> str:
    import torch

    cost = torch.rand(4096, 256) + 10.0
    cost[:, :8] = torch.rand(4096, 8)
    index, _ = constrained_assignment(cost, 1)
    used = int(torch.unique(index).numel())
    return "REPAIRED 8 -> " + str(used) if used == 256 else "FAIL used " + str(used)


def case_duplicate_codebook() -> str:
    import torch

    codes = torch.randn(256, 32)
    codes[1] = codes[0]
    try:
        verify_codebook(codes)
    except CodebookCollapse:
        return "REFUSED duplicate rows"
    return "FAIL not refused"


def case_tie_assignment() -> str:
    import torch

    cost = torch.full((4096, 256), 0.5)
    try:
        verify_margins(cost)
    except CodebookCollapse:
        return "REFUSED tie-decided"
    return "FAIL not refused"


def case_relative_spacing() -> str:
    import torch

    codes = torch.randn(64, 8)
    cost = torch.cdist(torch.randn(256, 8), codes)
    key = 424242
    verify_separation(codes, cost, key=key)
    collapsed = codes * 1e-3
    for _ in range(60):
        try:
            verify_separation(collapsed, cost, key=key)
        except CodebookCollapse:
            return "REFUSED spacing fell"
    return "FAIL not refused"


def case_bypass_forward(root: Path) -> str:
    tree = _fresh(root, "bypass")
    apply_floor(tree)
    module = tree / "model" / "layers.py"
    text = module.read_text(encoding="utf-8")
    module.write_text(text.replace("        embed_ind = self._indices(cost, infer_use_sk)",
                                   "        embed_ind = torch.argmin(cost, dim=-1)"),
                      encoding="utf-8")
    verdict = verify_tree(tree)
    return "REFUSED " + verdict.get("reason", "")[:40] if not verdict["ok"] else "FAIL passed"


def case_zero_floor(root: Path) -> str:
    tree = _fresh(root, "zero_floor")
    apply_floor(tree)
    module = tree / "model" / "layers.py"
    module.write_text(module.read_text(encoding="utf-8").replace(
        "ASSIGNMENT_FLOOR = 1", "ASSIGNMENT_FLOOR = 0"), encoding="utf-8")
    try:
        floor_constant(tree, "euclid")
    except UnknownTree as error:
        return "REFUSED " + str(error)[:38]
    return "FAIL not refused"


def case_residual_site(root: Path) -> str:
    tree = _fresh(root, "residual_site")
    apply_floor(tree)
    module = tree / "model" / "layers.py"
    module.write_text(module.read_text(encoding="utf-8")
                      + "\nclass Alt(VQLayer):\n    def pick(self, c):\n"
                        "        return c.min(dim=-1)[1]\n", encoding="utf-8")
    sites = unguarded_sites(tree, "euclid")
    return ("REFUSED " + sites[0]["file"] + ":" + str(sites[0]["line"])
            if sites else "FAIL not refused")


def case_soft_helper(root: Path) -> str:
    """A soft quantiser is protected when its argmax goes through the helper."""
    tree = _fresh(root, "soft_helper")
    module = tree / "model" / "layers.py"
    module.write_text(module.read_text(encoding="utf-8") + "\n" + "\n".join([
        "class SoftLayer(VQLayer):",
        "    def _indices(self, distances, infer_use_sk):",
        "        q = torch.softmax(-distances, dim=-1)",
        "        index, _ = constrained_assignment(-q, ASSIGNMENT_FLOOR)",
        "        return index.to(distances.device)",
    ]) + "\n", encoding="utf-8")
    apply_floor(tree)
    verdict = verify_tree(tree)
    return ("PROTECTED soft+helper" if verdict["ok"]
            else "FAIL " + verdict.get("reason", "")[:40])


def case_healthy(root: Path) -> str:
    tree = _fresh(root, "healthy")
    apply_floor(tree)
    verdict = verify_tree(tree)
    return ("PROTECTED " + "/".join(str(verdict[key]) for key in
                                    ("used", "used_balanced", "used_forward"))
            if verdict["ok"] else "FAIL " + verdict.get("reason", "")[:40])


CASES = (
    ("classic collapse", case_classic_collapse),
    ("duplicate codebook", case_duplicate_codebook),
    ("tie-decided assignment", case_tie_assignment),
    ("spacing fell 1000x", case_relative_spacing),
    ("bypass forward", case_bypass_forward),
    ("zero floor constant", case_zero_floor),
    ("residual selection site", case_residual_site),
    ("soft assignment via helper", case_soft_helper),
    ("healthy control", case_healthy),
)

EXPECTED = ("REPAIRED", "REFUSED", "REFUSED", "REFUSED", "REFUSED", "REFUSED", "REFUSED",
            "PROTECTED", "PROTECTED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path,
                        default=REPO / "results" / "research" / "qda" / "anticollapse_check")
    args = parser.parse_args(argv)
    root = args.tree.resolve()
    if not BASELINE.is_dir():
        raise ReadOnlyBaseline("the frozen baseline is missing: " + str(BASELINE))

    failures = 0
    width = max(len(name) for name, _ in CASES)
    for (name, case), expected in zip(CASES, EXPECTED):
        try:
            verdict = case(root) if case.__code__.co_argcount else case()
        except Exception as error:                                    # noqa: BLE001
            verdict = "ERROR " + type(error).__name__ + ": " + str(error)[:60]
        ok = verdict.startswith(expected)
        failures += 0 if ok else 1
        print(f"{name:<{width}}  {verdict:<42} {'ok' if ok else 'MISMATCH'}")
    shutil.rmtree(root, ignore_errors=True)
    print(f"\n{len(CASES) - failures}/{len(CASES)} cases as expected")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
