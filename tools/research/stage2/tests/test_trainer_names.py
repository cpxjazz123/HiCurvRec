"""Regression test for the export-time constant fault.

The fault: the accepted-state restore record used ``NUM_EPOCHS``, a constant the
trainer never defines (it defines ``EPOCHS``). The 3000-epoch run finished, the
final evaluation wrote its metrics, and then the export died with
``NameError: name 'NUM_EPOCHS' is not defined`` -- no ``item_sids.json``, and the
MCP reported the run as ``exited_incomplete`` after a complete training.

The guard is static: it reads every Stage 2 trainer's symbol table and refuses a
function that references a module-level name the module never binds. It carries a
self-test on the exact shape of the original fault.

Run:  python3 -m tools.research.stage2.tests.test_trainer_names
"""
from __future__ import annotations

import builtins
import symtable
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.stage2 import contract

_ALLOWED = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__builtins__", "__package__",
}


def find_undefined_globals(source: str) -> list[str]:
    """Module-level names a function references but the module never binds."""
    table = symtable.symtable(source, "<trainer>", "exec")
    defined = {symbol.get_name() for symbol in table.get_symbols()
               if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()}
    found: set[str] = set()

    def walk(t: symtable.SymbolTable) -> None:
        for symbol in t.get_symbols():
            if (t.get_type() == "function" and symbol.is_global()
                    and symbol.is_referenced()
                    and symbol.get_name() not in defined
                    and symbol.get_name() not in _ALLOWED):
                found.add(symbol.get_name())
        for child in t.get_children():
            walk(child)

    walk(table)
    return sorted(found)


def test_the_detector_catches_the_export_name_error() -> str:
    """The exact shape: a restore record naming NUM_EPOCHS."""
    source = """
EPOCHS = 3000


def main():
    _record("accepted_state_restored", epoch=EVAL_INTERVAL * (NUM_EPOCHS // EVAL_INTERVAL))
"""
    names = find_undefined_globals(source)
    assert "NUM_EPOCHS" in names, f"the detector missed the fault: {names}"
    return "the detector flags a reference to an undefined NUM_EPOCHS"


def test_no_trainer_references_an_undefined_global() -> str:
    """Every trainer must bind every module-level name it reads."""
    root = Path(contract.resolve_root()).resolve()
    trees = sorted((root / "stage2_RQ-VAE").glob("*/train_rqvae.py"))
    if not trees:
        return "no trainer to scan under stage2_RQ-VAE"
    offenders = []
    for path in trees:
        names = find_undefined_globals(path.read_text(encoding="utf-8"))
        if names:
            offenders.append(f"{path.relative_to(root)}: {names[:6]}")
    assert not offenders, "undefined module-level names: " + "; ".join(offenders[:6])
    return f"{len(trees)} trainers scanned, every global reference is bound"


def main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for test in tests:
        try:
            print("PASS", test.__name__ + ":", test(), flush=True)
        except AssertionError as exc:
            failures.append(test.__name__ + ": " + str(exc))
            print("FAIL", test.__name__ + ":", exc, flush=True)
        except Exception as exc:                               # noqa: BLE001
            failures.append(test.__name__ + ": " + repr(exc))
            print("ERROR", test.__name__ + ":", repr(exc), flush=True)
    if failures:
        print("TESTS_FAIL", failures, flush=True)
        return 1
    print("TESTS_PASS", len(tests), "trainer name invariants hold", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
