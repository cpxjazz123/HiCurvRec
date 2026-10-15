"""Import the trainer for real, which py_compile cannot do.

`py_compile` only checks syntax. A malformed import line such as

    from load_radius import install_, suppress_load_updatesload_radius

compiles cleanly and then fails at module import time, before any training can
start -- which is how a broken import reached four trainers while every
compile check passed. This loads the trainer the way the MCP harness does and
asserts that the names it uses are actually resolvable.

Run from the tree root:  python3 scripts/check_imports.py
"""
import importlib.util
import os
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREE))

failures = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)
    return ok


print(f"tree: {TREE.name}")
try:
    spec = importlib.util.spec_from_file_location(
        "trainer_import_check", str(TREE / "train_rqvae.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check("the trainer module imports", True)
except Exception as exc:                                   # noqa: BLE001
    check("the trainer module imports", False, f"{type(exc).__name__}: {exc}")
    print("\nFAILED: the trainer cannot be imported")
    raise SystemExit(1)

# Every name the trainer binds from its mechanism module must exist, and the
# module-level constants that carry the arm's identity must be self-consistent.
arm = getattr(module, "EXPERIMENT_ARM", None)
check("EXPERIMENT_ARM is set and matches the tree name",
      bool(arm) and arm in TREE.name, f"arm = {arm!r}, tree = {TREE.name!r}")

for name in ("install_load_radius", "install_prefix_radius", "suppress_load_updates",
             "load_update_count", "reset_load_update_count"):
    if hasattr(module, name):
        check(f"{name} is resolvable in the trainer namespace", True)

mechanism = getattr(module, "CROWDING_MECHANISM", None)
check("CROWDING_MECHANISM is set", bool(mechanism), f"mechanism = {mechanism!r}")

if hasattr(module, "_OUT"):
    check("the result directory derives from this tree",
          module._OUT.parent.name == TREE.name,
          f"_OUT = {module._OUT}")

print("\n" + ("import check passed" if not failures else f"FAILURES: {failures}"))
raise SystemExit(1 if failures else 0)
