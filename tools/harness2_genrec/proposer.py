"""The two proposers: the real one for the machine, the scripted one for the pipeline.

`OpenCodeProposer` is the real editor: OpenCode, in an isolated scratch workspace, with
the evidence on disk next to it. It replaces the Claude Code proposer that issue #101
asked for: issue #102 makes OpenCode the execution backend, and the production path must
not depend on Claude Code or Vertex credentials.

`ScriptedProposer` is the offline stand-in: no model, no network, deterministic edits.
It stays as the deterministic CI path.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from harness2.core.proposer import Proposer, Reply

from . import opencode

#: Never forwarded to a child session: the host's provider selection is ours, not theirs.
_DROP = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "CLOUD_ML_REGION",
)

_INVARIANT_PARAGRAPH = """
## Verification invariants

- A verifier COMPUTES the declared value from the fixture input. It never restates the
  declared value: a restated value agrees with itself and proves nothing.
- A verifier that cannot reproduce a fixture reports failure for that fixture. It never
  reports a success it did not compute.
"""

_DOMAIN_PARAGRAPH = """
## Fixture family: masked weighted sum

- The fixture kind is `masked_weighted_sum`: the value is the sum of `w * v` over the
  terms whose `mask` entry is truthy, and a masked-out term contributes nothing at all.
- The shipped verifier is `scripts/verify_masked_sum.py`; it takes exactly one fixture
  path and prints a `STATUS:` line and a `RESULT:` line.
"""

_SKILL_NAME = "mechanism_verification"
_SKILL_TRIGGER = (
    "- When a candidate harness must ship a verifier for a frozen fixture family, "
    f"load the `{_SKILL_NAME}` skill.\n"
)

_SKILL_BODY = f"""---
name: {_SKILL_NAME}
description: Use when a candidate harness must ship a verifier for a frozen fixture family - compute the declared value from the input, compare it, and report a failure for every fixture it cannot reproduce.
---

# Mechanism verification

1. Ship `scripts/verify_masked_sum.py`; it takes one fixture path as its only argument.
2. Compute the masked weighted sum from the fixture's `input` field.
3. Compare the computed value with the fixture's `expected` field within the tolerance
   and print `STATUS: PASS` or `STATUS: FAIL`, then `RESULT: <value>`.
4. Report failure for any fixture whose value you could not reproduce. A success you
   did not compute is worse than a failure.
"""

#: A verifier that computes the value. It contains no declared value: every number it
#: prints comes from the fixture.
_CORRECT_VERIFIER = '''#!/usr/bin/env python3
"""Reproduce the declared value of a masked-weighted-sum fixture."""

from __future__ import annotations

import json
import sys

TOLERANCE = 1e-9


def masked_weighted_sum(payload: dict) -> float:
    terms = payload["terms"]
    mask = payload.get("mask")
    if mask is None:
        mask = [1] * len(terms)
    if len(mask) != len(terms):
        raise ValueError("mask length does not match the term count")
    total = 0.0
    for term, keep in zip(terms, mask):
        if keep:
            total += float(term["w"]) * float(term["v"])
    return total


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_masked_sum.py <fixture.json>", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as handle:
        fixture = json.load(handle)
    value = masked_weighted_sum(fixture["input"])
    declared = float(fixture["expected"])
    ok = abs(value - declared) <= TOLERANCE
    print("STATUS: " + ("PASS" if ok else "FAIL"))
    print("RESULT: " + repr(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
'''

#: A verifier that restates a declared value. It always succeeds, including where it
#: cannot know the answer: this is the false-success shape the gate must reject.
_HARDCODED_VERIFIER = '''#!/usr/bin/env python3
"""Restate a declared value instead of computing it."""

from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: verify_masked_sum.py <fixture.json>", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as handle:
        json.load(handle)
    print("STATUS: PASS")
    print("RESULT: 1.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
'''


class GenrecProposer(Proposer):
    """Deprecated: the Claude Code proposer issue #101 asked for.

    Kept only so that an old configuration fails with a reason instead of silently
    running a different agent. Issue #102 makes OpenCode the execution backend.
    """

    def env(self) -> dict[str, str]:
        raise NotImplementedError(
            "the Claude Code proposer was retired by issue #102: use OpenCodeProposer. "
            "The production path must not depend on Claude Code or Vertex credentials."
        )


def _copy_context(source: Path, destination: Path) -> Dict[str, Any]:
    """Copy the evidence tree into the agent's workspace, file by file.

    The framework builds the evidence of the parallel candidates concurrently, and a
    directory that is being written while it is being read can lose a file between the
    directory listing and the copy. A file that vanished is recorded and the copy
    continues: the agent gets the evidence that exists, and the gap is in the audit
    rather than in a crash.
    """
    copied = 0
    missing: List[str] = []
    for path in sorted(Path(source).rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = Path(destination) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(path, target)
            copied += 1
        except FileNotFoundError:
            missing.append(relative.as_posix())
    return {"copied": copied, "missing": missing}


class OpenCodeProposer(Proposer):
    """The real editor: OpenCode, in an isolated scratch workspace.

    The evidence is copied into the workspace next to the agent, the agent edits
    ``context/harness/`` in place, and the edited tree comes back as the proposal. The
    full trajectory -- the event stream, the tool calls with their inputs and outputs,
    the cost, the tokens, the exit status -- is written to the audit directory, with
    every secret redacted.
    """

    def __init__(
        self,
        spec,
        audit_dir=None,
        *,
        model: str = "",
        timeout: int = 1800,
        repo_root: Path | None = None,
        use_mcp: bool = True,
    ) -> None:
        super().__init__(spec, audit_dir=audit_dir)
        self.model = model or opencode.model_name()
        self.timeout = int(timeout)
        self.repo_root = Path(repo_root) if repo_root else None
        self.use_mcp = bool(use_mcp)
        self._workspaces: List[Path] = []

    def cleanup(self) -> None:
        """Remove the scratch workspaces this proposer made."""
        for workspace in self._workspaces:
            shutil.rmtree(workspace, ignore_errors=True)
        self._workspaces = []

    def call(self, mission, context_dir, label, **kwargs) -> Reply:
        context = Path(context_dir)
        workspace = Path(tempfile.mkdtemp(prefix="h2-proposer-"))
        # The workspace holds the edited harness, and the caller reads it from the reply
        # after this returns, so it cannot be removed here. The previous one can.
        for previous in self._workspaces:
            shutil.rmtree(previous, ignore_errors=True)
        self._workspaces = [workspace]
        copied = _copy_context(context, workspace / "context")
        prompt = self._prompt(mission, kwargs.get("system", ""))
        mcp = None
        if self.use_mcp and self.repo_root is not None:
            mcp = opencode.mcp_server_block(self.repo_root)
        result = opencode.run_agent(
            workspace,
            prompt,
            model=self.model,
            timeout=kwargs.get("timeout") or self.timeout,
            mcp=mcp,
            title=label,
        )
        result["copied_context_files"] = copied
        self._audit(label, mission, result)
        harness = workspace / "context" / "harness"
        text = result.get("text") or ""
        if not text.strip():
            text = result.get("stdout") or ""
        reply = Reply(
            label=label,
            text=text,
            ok=not result.get("errors") and harness.is_dir(),
            cost_usd=float(result.get("cost") or 0.0),
            wall_s=float(result.get("elapsed_s") or 0.0),
            attempts=1,
            channel="opencode",
            workspace=harness if harness.is_dir() else None,
        )
        self._calls.append(reply)
        return reply

    def _prompt(self, mission: str, system: str) -> str:
        parts = []
        if system:
            parts.append(system)
        parts.append(str(mission))
        parts.append(
            "\n\n# HOW TO WORK IN THIS DIRECTORY\n\n"
            "The evidence is on disk under `context/`. Read it -- the mission points at the\n"
            "files. The harness you are editing is `context/harness/`: change those files in\n"
            "place with your normal tools, and create new ones where a new component is the\n"
            "right lever. Files you do not touch are inherited unchanged.\n\n"
            "When you are done, reply with ONE `<notes>` block and nothing else: for each\n"
            "change, the moment in the evidence that motivated it."
        )
        return "\n\n".join(parts)

    def _audit(self, label: str, mission: str, result: dict) -> None:
        if not self.audit_dir:
            return
        directory = Path(self.audit_dir)
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in label)
        (directory / f"{safe}.mission.md").write_text(str(mission), encoding="utf-8")
        (directory / f"{safe}.json").write_text(
            json.dumps(
                {
                    "label": label,
                    "model": result.get("model"),
                    "argv": result.get("argv"),
                    "exit_code": result.get("exit_code"),
                    "timed_out": result.get("timed_out"),
                    "elapsed_s": result.get("elapsed_s"),
                    "cost": result.get("cost"),
                    "tokens": result.get("tokens"),
                    "tool_calls": result.get("tool_calls"),
                    "errors": result.get("errors"),
                    "text": result.get("text"),
                    "stdout": result.get("stdout"),
                    "stderr": result.get("stderr"),
                    "workspace": result.get("workspace"),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )


class ScriptedProposer(Proposer):
    """A deterministic proposer: no model, no network, no credentials.

    The edits it makes are fixed by the candidate label, so a run is reproducible and a
    defect can be introduced on purpose: the first domain candidate ships a verifier
    that restates the declared value, which the adapter's static gate must reject.
    """

    def call(self, mission, context_dir, label, **kwargs) -> Reply:
        context = Path(context_dir)
        if label.startswith("select."):
            text = self._select(context)
        elif label.startswith("experience."):
            text = self._experience(context)
        else:
            workspace = self._propose(context, label.split(".")[0])
            reply = Reply(
                label=label,
                text="<notes>Added the verifier contract to the always-on protocol.</notes>",
                ok=True,
                attempts=1,
                workspace=workspace,
            )
            self._calls.append(reply)
            return reply
        reply = Reply(label=label, text=text, ok=True, attempts=1)
        self._calls.append(reply)
        return reply

    # -- the edits ------------------------------------------------------
    def _propose(self, context: Path, label: str) -> Path:
        harness = context / "harness"
        harness.mkdir(parents=True, exist_ok=True)
        claude = harness / "CLAUDE.md"
        body = claude.read_text(encoding="utf-8") if claude.is_file() else ""

        parts: list[str] = []
        if label.startswith("inv"):
            parts.append(_INVARIANT_PARAGRAPH)
        if label.startswith("dom"):
            parts.append(_DOMAIN_PARAGRAPH)
        if label.startswith("agg") or label.startswith("c"):
            parts.append(_INVARIANT_PARAGRAPH)
            parts.append(_DOMAIN_PARAGRAPH)

        if parts:
            claude.write_text(body.rstrip("\n") + "\n" + "\n".join(parts), encoding="utf-8")

        if label.startswith("dom"):
            skill = harness / ".agent" / "skills" / _SKILL_NAME / "SKILL.md"
            skill.parent.mkdir(parents=True, exist_ok=True)
            skill.write_text(_SKILL_BODY, encoding="utf-8")
            claude.write_text(
                claude.read_text(encoding="utf-8").rstrip("\n") + "\n" + _SKILL_TRIGGER,
                encoding="utf-8",
            )

        script = harness / "scripts" / "verify_masked_sum.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            _HARDCODED_VERIFIER if label == "dom1" else _CORRECT_VERIFIER, encoding="utf-8"
        )
        return harness

    # -- the blind selection --------------------------------------------
    def _select(self, context: Path) -> str:
        candidates = sorted((context / "candidates").glob("*.md"))
        if not candidates:
            return "```choice\nA\n```\n<why>no candidate produced an artifact</why>"
        best = None
        for path in candidates:
            text = path.read_text(encoding="utf-8")
            if "false successes: 0" in text and "reproduces the served fixture: yes" in text:
                best = path.stem
                break
        chosen = best or candidates[0].stem
        return (
            f"```choice\n{chosen}\n```\n"
            "<why>The candidate's verifier reproduces the served fixture, catches every "
            "held-out fixture whose value changed, and reports no false success. The "
            "others either could not reproduce the fixture or claimed a success they did "
            "not compute.</why>"
        )

    # -- the reflection --------------------------------------------------
    def _experience(self, context: Path) -> str:
        return (
            "<edits>\n"
            "component: scripts/verify_masked_sum.py\n"
            "change: compute the masked weighted sum from the fixture input and compare it "
            "with the declared value instead of restating the declared value\n"
            "observed: the held-out fixtures whose value changed are caught, and no fixture "
            "produced a false success\n"
            "verdict: helped\n"
            "</edits>\n\n"
            "<experience>\n"
            "cue: When a candidate harness must ship a verifier for a frozen fixture family\n"
            "lesson: Compute the value from the input; never restate the declared value, and "
            "never report a success you did not compute.\n"
            "evidence: The verifier caught every held-out fixture whose value changed and "
            "reported no false success, while the restating verifier was rejected by the "
            "static gate before it was ever rolled.\n"
            "confidence: high for this observed contrast\n"
            "</experience>"
        )


__all__ = ["GenrecProposer", "OpenCodeProposer", "ScriptedProposer"]
