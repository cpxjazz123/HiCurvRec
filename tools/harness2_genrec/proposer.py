"""The two proposers: the real one for the machine, the scripted one for the pipeline.

`GenrecProposer` is the framework's `Proposer` with one change: the host's provider
configuration wins. The framework's own `Proposer.env()` drops ``ANTHROPIC_BASE_URL`` /
``ANTHROPIC_AUTH_TOKEN`` and forces ``CLAUDE_CODE_USE_VERTEX=1`` (``core/proposer.py:46``
and ``core/proposer.py:105``); this machine has no Vertex credentials and a working
Anthropic-compatible endpoint instead. No credential is written to disk or to the
repository -- the environment is inherited, never recorded.

`ScriptedProposer` is the offline stand-in: no model, no network, deterministic edits.
It exists so the pipeline can be exercised end to end before a single token is spent,
and so the false-success rejection can be demonstrated on demand.
"""

from __future__ import annotations

import os
from pathlib import Path

from harness2.core.proposer import Proposer, Reply

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
    """The framework's proposer, pointed at the provider this machine actually has."""

    def env(self) -> dict[str, str]:
        e = {k: v for k, v in os.environ.items() if k not in _DROP}
        model = os.environ.get("HARNESS2_GENREC_MODEL", "").strip()
        if model:
            e["ANTHROPIC_MODEL"] = model
        return e


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


__all__ = ["GenrecProposer", "ScriptedProposer"]
