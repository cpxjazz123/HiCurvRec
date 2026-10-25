"""A lightweight, verified related-work pass (Issue #123).

The pass asks a scholarly API for the mechanism a proposal actually describes,
keeps the records it can verify, and says `NOT_CHECKED` when the network, the
rate limit or the API fails. It never invents a title, a DOI or a novelty claim:
"no search result" is not novelty, and a paper is only a match when its own
record says so.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OPENALEX = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/search"

#: The initial resource envelope for the pass.
MAX_QUERIES_PER_IDEA = 3
MAX_PAPERS_PER_IDEA = 15
REQUEST_TIMEOUT = 20

STATUS_OK = "OK"
STATUS_NOT_CHECKED = "NOT_CHECKED"
STATUS_UNAVAILABLE = "UNAVAILABLE"

#: How a retrieved paper relates to the proposal. Only the first is a genuine
#: collision; the others are context.
MATCH_SAME_OPERATOR = "same_operator_or_path"
MATCH_RELATED = "related_but_different"
MATCH_NOT_ESTABLISHED = "not_established"
MATCH_UNAVAILABLE = "literature_unavailable"


#: The keyword families a mechanism query is built from. A search string is a
#: mechanism, not a sentence: sending a whole paragraph to the API returns
#: unrelated work, which would look like evidence.
GEOMETRY_QUERY = "hyperbolic vector quantization"
BEHAVIOUR_QUERY = "user behavior conditioned codebook learning"
CURVATURE_QUERY = "adaptive curvature recommender semantic ID"
TITLE_QUERY = "hyperbolic quantization recommender"


def mechanism_queries(idea: dict) -> list:
    """Two or three searches that name the mechanism, not just its words."""
    text = " ".join(str(idea.get(field, "")) for field in
                    ("title", "key_change", "hyperbolic_operator", "behavior_signal",
                     "coupling_path")).lower()
    queries = []
    if any(token in text for token in ("hyperbolic", "poincare", "lorentz", "curvature")):
        queries.append(GEOMETRY_QUERY)
    if any(token in text for token in ("transition", "successor", "branch", "cohort",
                                       "behavior", "behaviour", "user")):
        queries.append(BEHAVIOUR_QUERY)
    if any(token in text for token in ("curvature", "adaptive", "geometry")):
        queries.append(CURVATURE_QUERY)
    if not queries:
        queries.append(TITLE_QUERY)
    return queries[:MAX_QUERIES_PER_IDEA]


def _cache_path(cache_dir: Path, query: str) -> Path:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:20]
    return Path(cache_dir) / ("openalex_" + digest + ".json")


def _fetch(query: str, *, cache_dir: Path, timeout: int = REQUEST_TIMEOUT) -> dict:
    """One query, with the on-disk cache and the fallback source."""
    cache = _cache_path(cache_dir, query)
    if cache.is_file():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            payload["from_cache"] = True
            return payload
        except (OSError, ValueError):
            pass
    url = (OPENALEX + "?" + urllib.parse.urlencode(
        {"search": query, "per-page": MAX_PAPERS_PER_IDEA, "mailto": ""}))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:                              # noqa: BLE001
        fallback = _fetch_semantic_scholar(query, timeout=timeout)
        if fallback is None:
            return {"status": STATUS_NOT_CHECKED, "query": query,
                    "reason": type(exc).__name__ + ": " + str(exc)[:180],
                    "source": "openalex", "results": []}
        return fallback
    records = []
    for work in payload.get("results", [])[:MAX_PAPERS_PER_IDEA]:
        record = _normalise(work)
        if record:
            records.append(record)
    result = {"status": STATUS_OK, "query": query, "source": "openalex",
              "retrieved_at": time.time(), "results": records,
              "total_reported": (payload.get("meta") or {}).get("count")}
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    except OSError:
        pass
    return result


def _fetch_semantic_scholar(query: str, *, timeout: int = REQUEST_TIMEOUT) -> dict:
    """The fallback source, used only when the primary one fails."""
    url = (SEMANTIC_SCHOLAR + "?" + urllib.parse.urlencode(
        {"query": query, "limit": MAX_PAPERS_PER_IDEA,
         "fields": "title,authors,year,externalIds,abstract,url"}))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:                                     # noqa: BLE001
        return None
    records = []
    for work in payload.get("data", [])[:MAX_PAPERS_PER_IDEA]:
        identifiers = work.get("externalIds") or {}
        record = {
            "title": work.get("title"),
            "authors": [author.get("name") for author in (work.get("authors") or [])
                        if author.get("name")][:8],
            "year": work.get("year"),
            "doi": identifiers.get("DOI"),
            "url": work.get("url") or (("https://doi.org/" + identifiers["DOI"])
                                       if identifiers.get("DOI") else None),
            "source_id": identifiers.get("CorpusId") or identifiers.get("DOI"),
            "abstract": (work.get("abstract") or "")[:1200],
            "source": "semantic_scholar",
        }
        if record["title"] and (record["doi"] or record["source_id"]):
            records.append(record)
    return {"status": STATUS_OK, "query": query, "source": "semantic_scholar",
            "retrieved_at": time.time(), "results": records}


def _normalise(work: dict) -> dict:
    """One OpenAlex work as a record with a verifiable identifier."""
    if not isinstance(work, dict):
        return None
    title = work.get("title") or work.get("display_name")
    identifier = work.get("doi") or work.get("id")
    if not title or not identifier:
        return None
    authors = []
    for author in (work.get("authorships") or [])[:8]:
        name = ((author or {}).get("author") or {}).get("display_name")
        if name:
            authors.append(name)
    abstract = work.get("abstract_inverted_index")
    text = ""
    if isinstance(abstract, dict) and abstract:
        positions = {}
        for word, places in abstract.items():
            for place in places:
                positions[place] = word
        text = " ".join(positions[key] for key in sorted(positions))[:1200]
    return {
        "title": title,
        "authors": authors,
        "year": work.get("publication_year"),
        "doi": work.get("doi"),
        "url": identifier if str(identifier).startswith("http") else ("https://doi.org/" + str(identifier) if work.get("doi") else identifier),
        "source_id": work.get("id"),
        "abstract": text,
        "source": "openalex",
        "cited_by": work.get("cited_by_count"),
    }


def classify(record: dict, idea: dict) -> str:
    """How the retrieved paper relates to the proposal's own mechanism.

    The comparison is against the whole mechanism the proposal states (its
    title, its curved operator, its behaviour signal and where they couple), not
    against one field: a paper can collide with the mechanism without repeating
    the proposal's own wording.
    """
    heading = (record.get("title") or "").lower()
    title = (heading + " " + (record.get("abstract") or "")).lower()
    proposal = " ".join(str(idea.get(field, "")) for field in
                         ("title", "key_change", "hyperbolic_operator", "behavior_signal",
                          "coupling_path", "coupling_equation")).lower()
    geometry = any(token in proposal for token in
                    ("hyperbolic", "poincare", "lorentz", "negative curvature", "curvature"))
    quantizer = any(token in proposal for token in
                    ("quantiz", "codebook", "discret", "semantic id"))
    paper_geometry = any(token in title for token in
                         ("hyperbolic", "poincare", "lorentz", "curvature"))
    # The paper has to be about the same OBJECT: a vector quantizer or a
    # recommender. A quantized adaptive filter merely mentions both words.
    paper_object = any(token in heading for token in
                       ("vector quantization", "vector quantiser", "codebook",
                        "discrete token", "semantic id", "recommender", "recommendation",
                        "recommend"))
    paper_object_anywhere = paper_object or any(
        token in title for token in ("vector quantization", "codebook", "semantic id"))
    if geometry and quantizer and paper_geometry and paper_object:
        return MATCH_SAME_OPERATOR
    if paper_geometry and paper_object_anywhere:
        return MATCH_RELATED
    if paper_object_anywhere:
        return MATCH_RELATED
    return MATCH_NOT_ESTABLISHED


def run_literature(idea: dict, *, cache_dir: Path, budget_seconds: float = 60.0) -> dict:
    """The related-work pass of one proposal."""
    started = time.monotonic()
    queries = mechanism_queries(idea)
    seen = {}
    outcomes = []
    status = STATUS_OK
    for query in queries:
        outcome = _fetch(query, cache_dir=Path(cache_dir))
        outcomes.append({"query": query, "status": outcome.get("status"),
                         "source": outcome.get("source"),
                         "results": len(outcome.get("results", [])),
                         "reason": outcome.get("reason")})
        if outcome.get("status") != STATUS_OK:
            status = STATUS_NOT_CHECKED
            continue
        for record in outcome.get("results", []):
            key = (record.get("doi") or record.get("source_id") or record.get("title"))
            if key in seen:
                continue
            record["match"] = classify(record, idea)
            seen[key] = record
    papers = list(seen.values())[:MAX_PAPERS_PER_IDEA]
    if not papers and status != STATUS_NOT_CHECKED:
        status = STATUS_UNAVAILABLE
    return {
        "schema": 1,
        "idea_id": idea.get("idea_id"),
        "status": status,
        "queries": outcomes,
        "papers": papers,
        "paper_count": len(papers),
        "retrieved_at": time.time(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limits": {"queries": MAX_QUERIES_PER_IDEA, "papers": MAX_PAPERS_PER_IDEA},
        "note": "a match is only a collision when the retrieved record itself concerns the "
                "mechanism; no result is never read as novelty",
    }
