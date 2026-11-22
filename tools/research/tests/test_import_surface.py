"""Every module of the research package must import on the interpreter that runs it.

The MCP server is started by the harness with the system python, and the system
python here is 3.9. A 3.10-only annotation - ``int | None`` without the future
import - raises at import time and takes the whole tool surface down at call
time, which is exactly what happened once. This test walks the package and
imports every module, so the failure appears in the suite instead of in a
research round.
"""
import importlib
import pkgutil
import unittest
from pathlib import Path

import tools.research

PACKAGE_ROOT = Path(tools.research.__file__).resolve().parent


def _module_names():
    names = []
    for module in pkgutil.walk_packages([str(PACKAGE_ROOT)], prefix="tools.research."):
        names.append(module.name)
    return sorted(names)


class ImportSurfaceTests(unittest.TestCase):
    def test_every_module_imports(self):
        failures = []
        for name in _module_names():
            try:
                importlib.import_module(name)
            except ModuleNotFoundError as exc:
                # A third-party dependency that this interpreter does not have
                # is not a compatibility problem.
                if exc.name and str(exc.name).split(".")[0] in (
                        "numpy", "pandas", "torch", "scipy", "sklearn", "yaml", "requests"):
                    continue
                failures.append((name, repr(exc)))
            except Exception as exc:                      # noqa: BLE001
                failures.append((name, type(exc).__name__ + ": " + str(exc)))
        self.assertEqual(failures, [], "modules that do not import: " + repr(failures))

    def test_the_package_declares_no_310_only_annotation(self):
        """A ``X | Y`` annotation outside a module with the future import."""
        offenders = []
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if "from __future__ import annotations" in source:
                continue
            for number, line in enumerate(source.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#") or "|" not in stripped:
                    continue
                if "->" in stripped or ":" in stripped:
                    if ("None" in stripped and "|" in stripped) or "Optional" in stripped:
                        offenders.append(str(path) + ":" + str(number) + ": " + stripped)
        self.assertEqual(offenders, [], "annotation that needs the future import: " +
                         repr(offenders))


if __name__ == "__main__":
    unittest.main()
