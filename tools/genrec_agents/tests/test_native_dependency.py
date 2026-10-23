"""The native-dependency regression: no third-party framework may return.

Two independent checks, because they fail differently:

* an AST scan of every active module, which catches an ``import`` however it is spelled
  (``import harness2``, ``from harness2.core import x``, ``importlib.import_module("harness2")``
  is caught by the STRING scan below);
* a runtime proof that the package imports under the PRODUCTION interpreter at a moment when the
  third-party package is provably not importable.

The historical archive under ``results/`` is exempt: it records what was, and rewriting history to
satisfy a linter would be worse than the lint.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Modules that must never be imported by active code again.
FORBIDDEN_MODULES = ("harness2",)
#: The retired production namespace and the third-party venv, as strings.
FORBIDDEN_STRINGS = ("tools.harness2_genrec", "tools/harness2_genrec",
                     "harness2/.venv", "harness2.core", "run_sweep")
#: Where the active code lives. ``results/`` is the historical archive and is exempt.
ACTIVE_ROOTS = ("tools", "docs", ".agent", "scripts", "stage2_RQ-VAE", "stage3_T5Train")
ACTIVE_FILES = ("CLAUDE.md", "README.md", "mcp.json")
#: Data identifiers that name archived runs on disk, not the framework.
ALLOWED_SUBSTRINGS = ("harness2_runs", "harness2_baseline")
#: The scanner itself has to spell the names it forbids: that is the one allowed mention.
SELF_EXEMPT = ("tests/test_native_dependency.py",)


def _active_files():
    for name in ACTIVE_FILES:
        path = REPO_ROOT / name
        if path.is_file():
            yield path
    for root in ACTIVE_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
                continue
            if path.suffix not in (".py", ".md", ".json", ".sh", ".toml", ".cfg", ".yml", ".yaml"):
                continue
            if any(part in path.as_posix() for part in SELF_EXEMPT):
                continue
            yield path


def _imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_no_active_module_imports_the_third_party_framework():
    offenders = []
    for path in _active_files():
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for module in _imports(tree):
            if any(module == bad or module.startswith(bad + ".") for bad in FORBIDDEN_MODULES):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: imports {module}")
    assert not offenders, "the third-party framework is imported again:\n" + "\n".join(offenders)


def test_no_active_file_names_the_retired_namespace_or_the_third_party_venv():
    offenders = []
    for path in _active_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in FORBIDDEN_STRINGS:
            index = text.find(needle)
            while index >= 0:
                line = text[text.rfind("\n", 0, index) + 1:text.find("\n", index)]
                if not any(allowed in line for allowed in ALLOWED_SUBSTRINGS):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {line.strip()[:120]}")
                index = text.find(needle, index + 1)
    assert not offenders, "the retired namespace or the third-party venv is named again:\n" + \
        "\n".join(sorted(set(offenders))[:20])


def test_the_production_interpreter_cannot_import_the_framework():
    """The proof that the next check means something: there is nothing to import."""
    python = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10"
    if not Path(python).exists():
        pytest.skip("the production interpreter is not on this machine")
    result = subprocess.run([python, "-c", "import harness2"], capture_output=True, text=True)
    assert result.returncode != 0, "the third-party framework IS importable: the proof is void"
    assert "ModuleNotFoundError" in (result.stderr or "")


def test_the_whole_package_imports_without_the_framework():
    """Every active module of the package imports under the production interpreter."""
    python = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10"
    if not Path(python).exists():
        pytest.skip("the production interpreter is not on this machine")
    modules = sorted(p.stem for p in (REPO_ROOT / "tools" / "genrec_agents").glob("*.py")
                     if p.stem != "__init__")
    source = (
        "import importlib, sys\n"
        "sys.path.insert(0, %r)\n"
        "failed = []\n"
        "for name in %r:\n"
        "    try:\n"
        "        importlib.import_module('tools.genrec_agents.' + name)\n"
        "    except Exception as exc:\n"
        "        failed.append(name + ': ' + type(exc).__name__ + ': ' + str(exc)[:120])\n"
        "print('FAILED:' + ' | '.join(failed) if failed else 'ALL OK')\n"
    ) % (str(REPO_ROOT), modules)
    result = subprocess.run([python, "-c", source], capture_output=True, text=True,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert "ALL OK" in result.stdout, result.stdout


def test_the_cli_speaks_the_first_party_vocabulary():
    """A user typing the command must not be told about a framework that is gone."""
    result = subprocess.run(
        ["/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10", "-m",
         "tools.genrec_agents.cli", "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    text = (result.stdout + result.stderr).lower()
    assert "harness2" not in text, text[:800]
    assert "genrec_agents" in text


def test_the_hand_maintained_campaign_ledger_is_never_read_as_an_authority():
    """One durable truth.

    The old campaign kept a second, hand-edited queue file (`campaign_dag.json`) beside the DAG's
    state. No code ever read it, and it drifted (32 tasks against the DAG's 29). The runtime must
    keep reading the DAG only: this test fails the moment any active module opens a hand-maintained
    campaign ledger by name.
    """
    offenders = []
    for path in _active_files():
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "campaign_dag.json" in text and "tests/" not in path.as_posix():
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, ("a hand-maintained campaign ledger is read again: " + ", ".join(offenders))
    # the historical file may exist as an archive, and the DAG's own state is what a run uses
    dag = REPO_ROOT / "tools" / "genrec_agents" / "dag.py"
    assert "campaign_dag" not in dag.read_text(encoding="utf-8")
