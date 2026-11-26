"""Tests for the Stage3 lightweight startup and dependency cache."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _startup():
    import tools.stage3.startup_cache as startup
    return startup


def _write_pkg(site: Path, name: str, payload: str):
    pkg = site / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(payload)
    (pkg / "data.txt").write_text("data:" + name)


def test_cache_reuses_same_source_and_rejects_changed_source(tmp_path):
    startup = _startup()
    site = tmp_path / "site"
    _write_pkg(site, "transformers", "TRANSFORMERS = 1\n")
    _write_pkg(site, "tokenizers", "TOKENIZERS = 1\n")
    cache = tmp_path / "cache"
    first = startup.prepare(site, cache, packages=("transformers", "tokenizers"))
    assert first is not None and (Path(first) / "transformers/__init__.py").is_file()
    second = startup.prepare(site, cache, packages=("transformers", "tokenizers"))
    assert str(first) == str(second)
    (site / "transformers/data.txt").write_text("changed")
    third = startup.prepare(site, cache, packages=("transformers", "tokenizers"))
    assert str(third) != str(first)
    assert (Path(third) / "transformers/data.txt").read_text() == "changed"


def test_cache_repairs_corrupt_target(tmp_path):
    startup = _startup()
    site = tmp_path / "site"
    _write_pkg(site, "transformers", "TRANSFORMERS = 2\n")
    cache = tmp_path / "cache"
    first = Path(startup.prepare(site, cache, packages=("transformers",)))
    (first / "transformers/data.txt").write_text("corrupt")
    repaired = Path(startup.prepare(site, cache, packages=("transformers",)))
    assert (repaired / "transformers/data.txt").read_text() == "data:transformers"


def test_missing_package_is_skipped_not_fabricated(tmp_path):
    startup = _startup()
    site = tmp_path / "site"
    _write_pkg(site, "transformers", "TRANSFORMERS = 3\n")
    result = startup.prepare(site, tmp_path / "cache",
                             packages=("transformers", "does_not_exist"))
    assert (Path(result) / "transformers/__init__.py").is_file()
    assert not (Path(result) / "does_not_exist").exists()


def test_generated_harness_is_lightweight_parent():
    spec = importlib.util.spec_from_file_location("shared_server", ROOT / "tools/genrec-mcp/server.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["shared_server"] = module
    spec.loader.exec_module(module)
    target = ROOT / "results/research/issue264/evidence/harness_lightweight.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    # The production writer has a fixed harness directory; inspect the template
    # contract here instead of writing into the live MCP harness directory.
    text = module.HARNESS_TEMPLATE
    assert "importlib.util.spec_from_file_location" in text
    assert "train_HG-Rec.py" in text
    before_parent = text.split("def _stage3_parent", 1)
    assert len(before_parent) == 2, "harness must separate a lightweight parent entry"
    parent = before_parent[1].split("def _stage3_worker", 1)[0]
    assert "spec_from_file_location" not in parent
    assert "exec_module" not in parent
    # The parent may start a separate prewarm process, but it must not import
    # the trainer or transformers itself.
    assert "exec_module" not in parent
    assert "trainer.main" not in parent
    assert "torchrun" in parent


def test_parent_branch_does_not_import_torch_and_propagates_exit(tmp_path, monkeypatch):
    """The parent branch is stdlib-only and returns the child's exit code."""
    import types
    from unittest import mock
    spec = importlib.util.spec_from_file_location("shared_server2", ROOT / "tools/genrec-mcp/server.py")
    server = importlib.util.module_from_spec(spec)
    sys.modules["shared_server2"] = server
    spec.loader.exec_module(server)
    server.HARNESS_DIR = tmp_path / "harness"
    harness = server.write_stage3_harness(tmp_path / "item_sids.json", "issue264_diag", False)
    code = compile(harness.read_text(), str(harness), "exec")
    module = types.ModuleType("issue264_harness")
    module.__file__ = str(harness)
    module.__name__ = "issue264_harness_worker"  # do not execute __main__
    exec(code, module.__dict__)
    calls = []
    def fake_call(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return 7
    monkeypatch.setenv("GENREC_STAGE3_MODE", "ddp")
    monkeypatch.setenv("GENREC_STAGE3_PREWARM", "0")
    monkeypatch.setattr("subprocess.call", fake_call)
    fake_cache = types.ModuleType("tools.stage3.startup_cache")
    fake_cache.prepare = lambda site: None
    monkeypatch.setitem(sys.modules, "tools.stage3.startup_cache", fake_cache)
    try:
        module._stage3_parent()
    except SystemExit as exc:
        assert exc.code == 7
    else:
        raise AssertionError("parent did not propagate the child exit code")
    assert calls and "--nproc_per_node=4" in calls[0][0]
    assert "torch" not in sys.modules


def test_parent_single_gpu_branch_preserves_single_flag(tmp_path, monkeypatch):
    import types
    spec = importlib.util.spec_from_file_location("shared_server3", ROOT / "tools/genrec-mcp/server.py")
    server = importlib.util.module_from_spec(spec)
    sys.modules["shared_server3"] = server
    spec.loader.exec_module(server)
    server.HARNESS_DIR = tmp_path / "harness"
    harness = server.write_stage3_harness(tmp_path / "item_sids.json", "issue264_diag", False)
    module = types.ModuleType("issue264_harness_single")
    module.__file__ = str(harness)
    module.__name__ = "issue264_harness_single_worker"
    exec(compile(harness.read_text(), str(harness), "exec"), module.__dict__)
    monkeypatch.setenv("GENREC_STAGE3_MODE", "single-gpu")
    monkeypatch.setenv("GENREC_STAGE3_PREWARM", "0")
    fake_cache = types.ModuleType("tools.stage3.startup_cache")
    fake_cache.prepare = lambda site: tmp_path / "cache"
    monkeypatch.setitem(sys.modules, "tools.stage3.startup_cache", fake_cache)
    envs = []
    monkeypatch.setattr("subprocess.call", lambda cmd, **kw: envs.append((cmd, kw["env"])) or 0)
    try:
        module._stage3_parent()
    except SystemExit as exc:
        assert exc.code == 0
    assert envs and envs[0][0][-1].endswith("run_stage3_param_issue264_diag.py")
    assert envs[0][1]["GENREC_SINGLE_GPU"] == "1"
    assert envs[0][1]["GENREC_STAGE3_LAUNCHED"] == "1"



def test_cache_is_atomic_under_concurrent_prepare(tmp_path):
    import subprocess
    site = tmp_path / "site"
    _write_pkg(site, "transformers", "TRANSFORMERS = 4\n")
    cache = tmp_path / "cache"
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from tools.stage3.startup_cache import prepare; "
        "print(prepare(%r, %r, packages=('transformers',)))"
        % (str(ROOT), str(site), str(cache))
    )
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for _ in range(2)]
    results = [p.communicate(timeout=60) for p in procs]
    assert all(p.returncode == 0 for p in procs), results
    targets = {out.strip().splitlines()[-1] for out, _ in results}
    assert len(targets) == 1
    target = Path(next(iter(targets)))
    assert (target / "transformers/__init__.py").read_text() == "TRANSFORMERS = 4\n"
    assert (target / "manifest.json").is_file()
