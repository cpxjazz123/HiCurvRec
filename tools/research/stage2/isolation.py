"""Refuse to start a diagnostic whose output paths escape the directory it was given.

Three times in this investigation a diagnostic run wrote into a *published* result
directory because a path constant is derived at import time (`EXPORT_MANIFEST`,
`item_sids.json`, the F0 observer's observation files): patching `RESULT_DIR` afterwards
does not move them. The mistake is silent -- the run looks fine and a previous artifact is
gone -- so it gets a check instead of a convention.

    python tools/research/stage2/isolation.py <tree> <allowed_root>

The tree's trainer is imported in a subprocess (import-time derivation is the whole point),
every `Path` named like an output target in every module loaded from the tree is resolved,
and any target outside `<allowed_root>` is reported. Exit 0 when all of them stay inside.

**Partial by construction.** It sees the paths that exist when the trainer imports, not the
ones a module computes later (the F0 observer builds its observation paths inside its own
functions, and that is the shape that clobbered `results/research/qda/runs/QDA_F0_seed42`
once). It also costs about forty seconds, because importing the trainer loads the config.
Use it as a first filter; the rule that actually holds is "list every path the run writes,
rebase each one, and check `git status` afterwards".
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROBE = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, {tree!r})
import train_rqvae as t

write_words = ("RESULT", "OUTPUT", "CHECKPOINT", "LOG_DIR", "METRICS", "MANIFEST",
               "OBSERVATION", "EXPORT", "SID")
targets = {{}}
# every module loaded from this tree, not just the trainer: the observer keeps its own
# output paths, and that is exactly where the third incident came from
for module_name, module in list(sys.modules.items()):
    location = getattr(module, "__file__", None)
    if not location or not str(Path(location).resolve()).startswith({tree_root!r}):
        continue
    for name, value in vars(module).items():
        if not isinstance(value, Path):
            continue
        if not any(word in name for word in write_words):
            continue
        targets[module_name.split(".")[-1] + "." + name] = str(value)
print(json.dumps(targets))
'''


def outside_root(targets: dict[str, str], allowed: Path) -> dict[str, str]:
    """The classification on its own: which named targets resolve outside `allowed`."""
    allowed = Path(allowed).resolve()
    return {name: value for name, value in targets.items()
            if not Path(value).resolve().is_relative_to(allowed)}


def check(tree: Path, allowed: Path) -> dict:
    resolved = str(tree.resolve())
    script = PROBE.format(tree=resolved, tree_root=resolved)
    done = subprocess.run([sys.executable, "-c", script], cwd=str(tree),
                          capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        return {"status": "FAIL", "detail": "the trainer could not be imported: "
                + done.stderr.strip().splitlines()[-1][:200], "targets": {}}
    targets = json.loads(done.stdout.strip().splitlines()[-1])
    allowed = allowed.resolve()
    outside = outside_root(targets, allowed)
    if outside:
        return {"status": "FAIL", "allowed": str(allowed), "outside": outside,
                "detail": "output paths escape " + str(allowed)}
    return {"status": "PASS", "allowed": str(allowed), "targets": targets}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tree", type=Path)
    parser.add_argument("allowed_root", type=Path)
    args = parser.parse_args(argv)
    report = check(args.tree, args.allowed_root)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
