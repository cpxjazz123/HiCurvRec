"""Idea records: schema, cross-agent duplicates and provenance (Issue #120).

An idea is what one ideation agent proposes. It must be minimally specified
enough to be implemented, but it does not have to carry a proof: the issue
explicitly allows unconventional and unfinished ideas and rejects only the ones
that cannot be made minimally specified.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable

from . import scope as scope_module

#: Every idea carries these fields. ``novelty`` is relative to the mechanisms
#: that already exist in the repository, not a claim of priority.
REQUIRED_FIELDS = (
    "idea_id", "title", "motivation", "novelty", "key_change", "specification",
    "cost_estimate", "falsifiable_hypothesis", "risks",
) + scope_module.REQUIRED_SCOPE_FIELDS

MAX_IDEA_ID_LEN = 64
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

#: The structural maximum of one round: five ideation agents, three ideas each.
#: A cap below that would kill a round precisely when the agents are productive,
#: so the bound is the structure, not a round-specific number.
MAX_IDEAS_PER_ROUND = 15

#: A round needs three distinct proposals before the scorers can compare them.
#: The ideation phase asks the agents for one more idea each when they agree
#: with each other and produce fewer than this.
MIN_ROUND_IDEAS = 3


class IdeaError(ValueError):
    """An idea cannot be made minimally specified."""


def validate_idea(idea: dict, *, agent: str = "") -> dict:
    """Check one idea and return it with its provenance attached."""
    if not isinstance(idea, dict):
        raise IdeaError("idea must be an object")
    missing = [field for field in REQUIRED_FIELDS
               if not str(idea.get(field, "")).strip()]
    if missing:
        raise IdeaError("idea is missing: " + ", ".join(missing))
    identifier = str(idea["idea_id"])
    if not _ID_RE.match(identifier):
        raise IdeaError("idea_id must be a short safe identifier: " + identifier)
    record = dict(idea)
    record["idea_id"] = identifier
    record["origin_agent"] = agent or str(idea.get("origin_agent", ""))
    # The scope gate is a check on the content the agent wrote, not on a flag
    # it set: an idea that does not couple behaviour and geometry is rejected
    # with the reasons, and it is never reinterpreted as acceptable.
    verdict = scope_module.check_scope(record)
    record["scope_verdict"] = "IN_SCOPE" if verdict["in_scope"] else "OUT_OF_SCOPE"
    record["scope_reasons"] = verdict["reasons"]
    if not verdict["in_scope"]:
        raise IdeaError("out of scope: " + "; ".join(verdict["reasons"][:6]))
    if isinstance(idea.get("risks"), str):
        record["risks"] = [idea["risks"]]
    if isinstance(idea.get("specification"), str):
        record["specification"] = {"text": idea["specification"]}
    return record


def idea_fingerprint(idea: dict) -> str:
    """A stable identity for near-duplicate detection.

    Two agents proposing the same mechanism with different words must collide, so
    the fingerprint uses the numeric and mathematical content of the idea rather
    than its prose.
    """
    payload = {
        "title": _normalise(str(idea.get("title", ""))),
        "key_change": _normalise(str(idea.get("key_change", ""))),
        "specification": _normalise(json.dumps(idea.get("specification", ""), sort_keys=True,
                                               ensure_ascii=False)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _normalise(text: str) -> str:
    lowered = str(text).lower()
    numbers = re.findall(r"\d+(?:\.\d+)?", lowered)
    words = [word for word in re.findall(r"[a-z_]{4,}", lowered)]
    return "|".join(sorted(set(numbers)) + sorted(set(words)))


def collect_ideas(per_agent: Iterable[dict], require_minimum: bool = True) -> list:
    """Merge the ideas of several agents, keeping every origin.

    A near-duplicate is recorded as a duplicate of the idea it repeats, with the
    provenance of every agent that proposed it; the first occurrence stays the
    canonical record and the later ones are listed under it.
    """
    canonical: dict = {}
    order: list = []
    duplicates: list = []
    collected_rejections: list = []
    used_ids: dict = {}
    for agent, payload in per_agent:
        ideas = payload.get("ideas") if isinstance(payload, dict) else payload
        if not isinstance(ideas, list) or not ideas:
            raise IdeaError("agent " + str(agent) + " returned no ideas")
        if len(ideas) > 3:
            raise IdeaError("agent " + str(agent) + " returned more than three ideas")
        rejected = []
        for raw in ideas:
            try:
                idea = validate_idea(raw, agent=str(agent))
            except IdeaError as exc:
                # One malformed idea does not cost the agent its other ideas.
                rejected.append({"idea_id": str(raw.get("idea_id", "")) if isinstance(raw, dict) else "",
                                 "origin_agent": str(agent), "reason": str(exc)})
                continue
            # An identifier has to name one mechanism inside the round: two
            # agents that pick the same name would otherwise make the scorer's
            # two legitimate judgements look like a broken rubric.
            original_id = idea["idea_id"]
            if original_id in used_ids:
                suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(agent))
                renamed = original_id + "__" + suffix
                while renamed in used_ids:
                    renamed = renamed + "_x"
                idea["idea_id"] = renamed
                idea["renamed_from"] = original_id
                used_ids[renamed] = True
            else:
                used_ids[original_id] = True
            fingerprint = idea_fingerprint(idea)
            if fingerprint in canonical:
                canonical[fingerprint]["also_proposed_by"].append(idea["origin_agent"])
                duplicates.append({"idea_id": idea["idea_id"], "origin_agent": idea["origin_agent"],
                                   "duplicate_of": canonical[fingerprint]["idea_id"],
                                   "fingerprint": fingerprint})
                continue
            idea["fingerprint"] = fingerprint
            idea["also_proposed_by"] = [idea["origin_agent"]]
            canonical[fingerprint] = idea
            order.append(fingerprint)
        if rejected:
            collected_rejections.extend(rejected)
    merged = [canonical[key] for key in order]
    if require_minimum and len(merged) < MIN_ROUND_IDEAS:
        raise IdeaError("a round needs at least " + str(MIN_ROUND_IDEAS) +
                        " distinct ideas, got " + str(len(merged)))
    if len(merged) > MAX_IDEAS_PER_ROUND:
        raise IdeaError("a round may not hold more than " + str(MAX_IDEAS_PER_ROUND) +
                        " ideas, got " + str(len(merged)))
    for idea in merged:
        idea["duplicate_count"] = len(idea["also_proposed_by"])
    if collected_rejections:
        merged[0].setdefault("rejected_ideas", []).extend(collected_rejections)
    return merged


def idea_index(ideas: list) -> dict:
    return {idea["idea_id"]: idea for idea in ideas}
