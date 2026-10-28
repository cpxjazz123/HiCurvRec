"""Regression tests for the agent output contract.

The failures these reproduce are the ones this programme actually hit: a
cancelled reviewer whose partial text was read as a verdict, an expired
credential retried until the retry budget was gone, and one crashed reviewer
throwing away the four that had answered.

They run with the system interpreter and a fake agent binary: no model, no
network, no torch.

Run:  python3 -m tools.research.stage2.tests.test_agent_contract
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research import settings as research_settings
from tools.research.agents import runner
from tools.research.rounds import plaintext, worker


def _fake_agent(directory: Path, body: str) -> Path:
    path = directory / "fake-opencode"
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _prompt_and_workspace(root: Path, task_id: str) -> str:
    destination = root / "agents" / task_id / "verification.txt"
    return "VERDICT test.\n\nWrite the answer to " + str(destination) + " (this absolute path)."


def test_partial_answer_is_not_published() -> str:
    """A run that times out keeps its text as evidence, not as the answer."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        binary = _fake_agent(root, 'printf "VERDICT: PASS\\n" > verification.txt\n'
                                   'sleep 30\n')
        old_binary, old_timeout = research_settings.AGENT_BINARY, research_settings.AGENT_TIMEOUT_SECONDS
        research_settings.AGENT_BINARY = str(binary)
        research_settings.AGENT_TIMEOUT_SECONDS = 2
        try:
            (root / "agents" / "t1").mkdir(parents=True)
            destination = root / "agents" / "t1" / "verification.txt"
            try:
                runner.run_agent("t1", "prompt", "verification.txt", root,
                                 binary=str(binary), timeout=2, role="verify")
            except runner.AgentFailure as exc:
                assert exc.failure_class == "timeout", exc.failure_class
            else:
                raise AssertionError("a timed-out run did not fail")
            assert not destination.exists(), "the partial answer stayed where the cache reads"
            receipt = json.loads((root / "agents" / "t1" / "execution.json").read_text())
            assert receipt["verified"] is False, "a timed-out run was recorded as verified"
            assert receipt["failure_class"] == "timeout", receipt["failure_class"]
        finally:
            research_settings.AGENT_BINARY, research_settings.AGENT_TIMEOUT_SECONDS = old_binary, old_timeout
    return "a timed-out run leaves evidence and no answer"


def test_expired_credential_is_classified() -> str:
    """An authentication failure is its own class, not a generic crash."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        binary = _fake_agent(root, 'echo \'{"type":"error","error":{"type":"provider.auth",'
                                    '"message":"Your authentication token has expired."}}\'\n'
                                    'exit 1\n')
        try:
            runner.run_agent("t2", "prompt", "verification.txt", root,
                             binary=str(binary), timeout=30, role="verify")
        except runner.AgentFailure as exc:
            assert exc.failure_class == "auth", exc.failure_class
        else:
            raise AssertionError("a failed run did not fail")
    return "an expired credential is classified as auth"


def test_cache_requires_a_verified_receipt() -> str:
    """An answer on disk is reused only when its receipt says the run succeeded."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        workspace = root / "agents" / "t3"
        workspace.mkdir(parents=True)
        answer = workspace / "verification.txt"
        answer.write_text("VERDICT: PASS\nCHECK VERDICT: PASS\n")

        # (a) no receipt: not reusable
        binary = _fake_agent(root, 'printf "VERDICT: PASS\\nCHECK VERDICT: PASS\\n" > verification.txt\n')
        old_binary = research_settings.AGENT_BINARY
        research_settings.AGENT_BINARY = str(binary)
        try:
            report = worker._run_agent(root, "t3", "prompt", "verification.txt",
                                       plaintext.parse_verification, role="verify")
        finally:
            research_settings.AGENT_BINARY = old_binary
        assert report["verdict"] == "PASS", "the agent's answer was not parsed"
        assert (workspace / "unverified").is_dir(), \
            "the unverified answer was not moved aside before the agent ran again"

        # (b) a verified receipt: reusable without running the agent
        receipt = json.loads((workspace / "execution.json").read_text())
        assert receipt["verified"] is True, "a valid run was recorded as unverified"
        research_settings.AGENT_BINARY = str(root / "does-not-exist")
        try:
            again = worker._run_agent(root, "t3", "prompt", "verification.txt",
                                      plaintext.parse_verification, role="verify")
        finally:
            research_settings.AGENT_BINARY = old_binary
        assert again["verdict"] == "PASS", "a verified answer was not reused"
    return "the cache needs a verified receipt and re-runs otherwise"


def test_one_failed_agent_keeps_the_others() -> str:
    """Four answers survive a fifth agent that crashes."""
    def worker_task(index):
        if index == 3:
            raise runner.AgentFailure("auth", "receipt", "no answer")
        return {"index": index}

    try:
        worker._run_agents_concurrently([(index,) for index in range(1, 6)], worker_task)
    except worker.AgentPhaseIncomplete as exc:
        assert len(exc.results) == 4, "the answers of the healthy agents were discarded"
        assert [entry["failure_class"] for entry in exc.failures] == ["auth"]
        assert exc.classes == {"auth"}
        return "four answers are kept when the fifth fails"
    raise AssertionError("a failed agent did not mark the phase incomplete")


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
        except Exception as exc:                              # noqa: BLE001
            failures.append(test.__name__ + ": " + repr(exc))
            print("ERROR", test.__name__ + ":", repr(exc), flush=True)
    if failures:
        print("TESTS_FAIL", failures, flush=True)
        return 1
    print("TESTS_PASS", len(tests), "agent contract invariants hold", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
