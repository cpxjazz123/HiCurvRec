#!/usr/bin/env python3
"""Per-arm evidence must never share an output file.

The evaluator used to write every arm's result into one ``arm_alignment.json`` and to key
the arm under the fixed name ``chain_align``.  A later invocation for a different arm that
had no SID yet therefore replaced the finished arm's evidence with ``{"missing": ...}``,
and the numbers a run was judged by could not be recovered from disk.  These checks pin the
two properties that prevent it: an arm-named key and an arm-named default path.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_arm as evaluator  # noqa: E402


def test_default_output_is_per_arm() -> None:
    first = evaluator.output_path("TIGER_chain_align_rq")
    second = evaluator.output_path("TIGER_graphlet_align_rq")
    assert first != second, "two arms must not share one evidence file"
    assert first.name == "arm_alignment_TIGER_chain_align_rq.json"
    assert second.name == "arm_alignment_TIGER_graphlet_align_rq.json"


def test_report_key_is_the_arm_name() -> None:
    assert evaluator.arm_key("TIGER_chain_align_rq") == "TIGER_chain_align_rq"
    assert evaluator.arm_key("TIGER_graphlet_align_rq") != "chain_align"


def test_writing_one_arm_leaves_the_other_intact(tmp_path: Path) -> None:
    first = tmp_path / evaluator.output_path("TIGER_chain_align_rq").name
    second = tmp_path / evaluator.output_path("TIGER_graphlet_align_rq").name
    evaluator.write_report(first, {"TIGER_chain_align_rq": {"collision": 0.15}})
    evaluator.write_report(second, {"TIGER_graphlet_align_rq": {"missing": "/x"}})
    assert "TIGER_chain_align_rq" in first.read_text()
    assert "TIGER_chain_align_rq" not in second.read_text()


if __name__ == "__main__":
    test_default_output_is_per_arm()
    test_report_key_is_the_arm_name()
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_writing_one_arm_leaves_the_other_intact(Path(directory))
    print("ok")
