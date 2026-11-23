"""v2 join / manifest acceptance tests (issue #246 C28-C29, C34)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.graph_infra import assets, config  # noqa: E402

REPO = config.REPO


def test_join_rejects_every_bad_input():
    bank = np.array([[0, 1], [2, 3]], dtype=np.int64)
    good_scores = np.array([0.5, 0.7])
    # shape
    try:
        assets.join_scores_v2(bank, np.array([[0, 1]]), np.array([0.5]),
                              expected_graph_version="v2/test", bank_graph_version="v2/test")
        raise AssertionError("shape mismatch must raise")
    except ValueError:
        pass
    # dtype
    try:
        assets.join_scores_v2(bank, bank.astype(np.float64), good_scores,
                              expected_graph_version="v2/test", bank_graph_version="v2/test")
        raise AssertionError("float pair ids must raise")
    except ValueError:
        pass
    # duplicate
    try:
        assets.join_scores_v2(bank, np.array([[0, 1], [0, 1]]), good_scores,
                              expected_graph_version="v2/test", bank_graph_version="v2/test")
        raise AssertionError("duplicate model pairs must raise")
    except ValueError:
        pass
    # missing
    try:
        assets.join_scores_v2(bank, np.array([[0, 1], [9, 9]]), good_scores,
                              expected_graph_version="v2/test", bank_graph_version="v2/test")
        raise AssertionError("missing bank pair must raise")
    except ValueError:
        pass
    # extra
    try:
        assets.join_scores_v2(bank, np.array([[0, 1], [2, 3], [4, 5]]),
                              np.array([0.5, 0.7, 0.9]), expected_graph_version="v2/test", bank_graph_version="v2/test")
        raise AssertionError("extra model pair must raise")
    except ValueError:
        pass
    # non-finite
    try:
        assets.join_scores_v2(bank, bank, np.array([0.5, np.nan]), expected_graph_version="v2/test", bank_graph_version="v2/test")
        raise AssertionError("non-finite score must raise")
    except ValueError:
        pass
    # graph version
    try:
        assets.join_scores_v2(bank, bank, good_scores, expected_graph_version="v1/other",
                              bank_graph_version="v2/test")
        raise AssertionError("graph version mismatch must raise")
    except ValueError:
        pass


def test_join_accepts_a_reorder_by_pair_id():
    bank = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    shuffled_pairs = np.array([[4, 5], [0, 1], [2, 3]], dtype=np.int64)
    shuffled_scores = np.array([0.9, 0.5, 0.7])
    aligned = assets.join_scores_v2(bank, shuffled_pairs, shuffled_scores,
                                    expected_graph_version="v2/test", bank_graph_version="v2/test")
    assert aligned.tolist() == [0.5, 0.7, 0.9]


def _manifest(tmp_path, entries):
    path = tmp_path / "manifest.json"
    payload = {"issue": 246, "version": "v2", "assets": entries}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_requires_portable_paths_and_the_required_fields():
    result = assets.audit_manifest(REPO / "results" / "research" / "infra245" / "assets" / "manifest.json")
    assert result["ok"] is False or result["issues"], "a v1 manifest must not pass the v2 schema"
    required = {"name", "path", "schema", "graph_version", "legal_use", "version", "purpose",
                "command", "source_sha256", "code_sha256", "dependencies", "sha256", "bytes"}
    v2 = REPO / "results" / "research" / "infra246" / "assets" / "manifest.json"
    if v2.exists():
        payload = json.loads(v2.read_text())
        for entry in payload["assets"]:
            assert required <= set(entry), (entry["name"], required - set(entry))
            assert not Path(entry["path"]).is_absolute(), entry["path"]


def test_legal_use_is_enforced(tmp_path):
    target = tmp_path / "asset.bin"
    target.write_bytes(b"x")
    entry = assets.entry_v2("demo", target, repo=tmp_path, schema="bin", shape=[1], dtype="uint8",
                            graph_version="v2/full", legal_use="description", version="v2",
                            purpose="demo", command="echo", source_sha256="00",
                            code_sha256="11", dependencies=[])
    path = _manifest(tmp_path, [entry])
    assets.assert_legal_use(path, "demo", "description")
    try:
        assets.assert_legal_use(path, "demo", "generalisation")
        raise AssertionError("a description-only asset must not be used for generalisation")
    except ValueError:
        pass


def test_verify_does_not_write_and_keeps_the_expectation(tmp_path):
    target = tmp_path / "asset.bin"
    target.write_bytes(b"x")
    entry = assets.entry_v2("demo", target, repo=tmp_path, schema="bin", shape=[1], dtype="uint8",
                            graph_version="v2/full", legal_use="description", version="v2",
                            purpose="demo", command="echo", source_sha256="00",
                            code_sha256="11", dependencies=[])
    path = _manifest(tmp_path, [entry])
    before = path.read_bytes()
    assets.verify_manifest(path)
    assert path.read_bytes() == before
    target.write_bytes(b"tampered")
    try:
        assets.verify_manifest(path)
        raise AssertionError("drift must raise")
    except ValueError:
        pass
    assert json.loads(path.read_text())["assets"][0]["sha256"] == entry["sha256"], "no auto rebuild"


def main() -> int:
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)
             and value.__code__.co_argcount == 0]
    failures = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001
            failures.append((name, str(error)))
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
