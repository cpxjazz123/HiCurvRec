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

#: Every idea carries these fields. ``novelty`` is relative to the mechanisms
#: that already exist in the repository, not a claim of priority.
REQUIRED_FIELDS = (
    "idea_id", "title", "motivation", "novelty", "key_change", "specification",
    "cost_estimate", "falsifiable_hypothesis", "risks",
)

MAX_IDEA_ID_LEN = 64
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


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


def collect_ideas(per_agent: Iterable[dict]) -> list:
    """Merge the ideas of several agents, keeping every origin.

    A near-duplicate is recorded as a duplicate of the idea it repeats, with the
    provenance of every agent that proposed it; the first occurrence stays the
    canonical record and the later ones are listed under it.
    """
    canonical: dict = {}
    order: list = []
    duplicates: list = []
    for agent, payload in per_agent:
        ideas = payload.get("ideas") if isinstance(payload, dict) else payload
        if not isinstance(ideas, list) or not ideas:
            raise IdeaError("agent " + str(agent) + " returned no ideas")
        if len(ideas) > 3:
            raise IdeaError("agent " + str(agent) + " returned more than three ideas")
        for raw in ideas:
            idea = validate_idea(raw, agent=str(agent))
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
    merged = [canonical[key] for key in order]
    if len(merged) < 3:
        raise IdeaError("a round needs at least three distinct ideas, got " + str(len(merged)))
    if len(merged) > 9:
        raise IdeaError("a round may not hold more than nine ideas, got " + str(len(merged)))
    for idea in merged:
        idea["duplicate_count"] = len(idea["also_proposed_by"])
    return merged


def idea_index(ideas: list) -> dict:
    return {idea["idea_id"]: idea for idea in ideas}
