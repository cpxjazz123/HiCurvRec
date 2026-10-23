"""Agent L -- the literature research agent (issue #110, WP1).

The literature agent grounds hyperbolic ideas and novelty claims in REAL external papers. The
rule it works under is short: a search is a request and a response that are both recorded, a
citation that does not resolve is a failure of that claim, and when no external source is
reachable the agent persists ``LITERATURE_BLOCKED`` rather than inventing novelty.

The providers are explicit: a provider is a named source with a version, it is asked a
question, and it answers with records. Nothing claims to have searched without a request and
a response behind it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where the literature agent writes its four files.
LITERATURE_DIR = REPO_ROOT / "results" / "hyperbolic_research_runs" / "literature"

#: The evidence levels a paper record may carry.
EVIDENCE_LEVELS = (
    "verified_fulltext",
    "verified_abstract_only",
    "disputed",
    "unavailable",
)

#: The state the subsystem persists when no external source is reachable.
LITERATURE_BLOCKED = "LITERATURE_BLOCKED"

#: The user agent the providers send.
USER_AGENT = "HiCurvRec-hyperbolic-research/1.0 (issue #110)"


class LiteratureBlocked(RuntimeError):
    """No external source is reachable, so no novelty claim may be made."""


@dataclass
class PaperRecord:
    """One paper, as far as it has been verified."""

    identifier: str                       # the DOI or the arXiv id
    title: str = ""
    authors: Tuple[str, ...] = ()
    year: Optional[int] = None
    venue: str = ""
    url: str = ""
    source: str = ""
    access_level: str = "unavailable"
    retrieved_utc: str = ""
    abstract: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def identity(self) -> str:
        """The paper's identity, for deduplication: its identifier, lower-cased."""
        return (self.identifier or self.title).strip().lower()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "source": self.source,
            "access_level": self.access_level,
            "retrieved_utc": self.retrieved_utc,
            "abstract": self.abstract,
            "sections": dict(self.sections),
            "notes": self.notes,
        }


@dataclass
class SearchResponse:
    """What a provider answered, and the request it answered."""

    provider: str
    version: str
    query: str
    requested_utc: str
    records: List[PaperRecord] = field(default_factory=list)
    error: str = ""
    request_url: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "version": self.version,
            "query": self.query,
            "requested_utc": self.requested_utc,
            "count": len(self.records),
            "error": self.error,
            "request_url": self.request_url,
            "identifiers": [record.identifier for record in self.records],
        }


class Provider(Protocol):
    """A named academic source that can be asked a question."""

    name: str
    version: str

    def search(self, query: str, *, limit: int = 10) -> SearchResponse:  # pragma: no cover
        ...


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_json(url: str, *, timeout: int = 30) -> Tuple[Optional[Any], str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            return json.loads(handle.read().decode("utf-8")), ""
    except urllib.error.HTTPError as error:
        return None, f"HTTP {error.code}"
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}"


def _http_text(url: str, *, timeout: int = 30) -> Tuple[Optional[str], str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            return handle.read().decode("utf-8", errors="replace"), ""
    except urllib.error.HTTPError as error:
        return None, f"HTTP {error.code}"
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}"


class CrossrefProvider:
    """The Crossref REST API: a DOI, a title, the authors and the year."""

    name = "crossref"
    version = "rest-v1"

    def search(self, query: str, *, limit: int = 10) -> SearchResponse:
        response = SearchResponse(
            provider=self.name,
            version=self.version,
            query=query,
            requested_utc=now_utc(),
        )
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
            {"query": query, "rows": max(int(limit), 1), "select": "DOI,title,author,issued,container-title,URL,abstract"}
        )
        response.request_url = url
        document, error = _http_json(url)
        if error or not isinstance(document, dict):
            response.error = error or "the response was not an object"
            return response
        for item in (document.get("message") or {}).get("items") or ():
            titles = item.get("title") or []
            title = titles[0] if titles else ""
            issued = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
            authors = tuple(
                " ".join(part for part in (a.get("given"), a.get("family")) if part)
                for a in (item.get("author") or ())
            )
            venue = (item.get("container-title") or [""])[0]
            response.records.append(
                PaperRecord(
                    identifier=str(item.get("DOI") or ""),
                    title=title,
                    authors=authors,
                    year=int(issued[0]) if issued and issued[0] else None,
                    venue=venue,
                    url=str(item.get("URL") or ""),
                    source=self.name,
                    access_level="verified_abstract_only",
                    retrieved_utc=response.requested_utc,
                    abstract=_strip_tags(str(item.get("abstract") or "")),
                )
            )
        return response


def _openalex_abstract(item: Dict[str, Any]) -> str:
    """Rebuild the abstract from OpenAlex's inverted index.

    OpenAlex stores an abstract as ``word -> [positions]``; the text is the words in
    position order. A record without an inverted index has no abstract to read, and saying
    so is the honest level for it.
    """
    index = item.get("abstract_inverted_index") or {}
    if not isinstance(index, dict) or not index:
        return ""
    positions: List[Tuple[int, str]] = []
    for word, slots in index.items():
        for slot in slots or ():
            try:
                positions.append((int(slot), str(word)))
            except (TypeError, ValueError):
                continue
    positions.sort()
    return " ".join(word for _, word in positions)


class OpenAlexProvider:
    """The OpenAlex works API: a DOI, a title, the authors, the year and the abstract.

    OpenAlex indexes arXiv-only preprints under the DOI ``10.48550/arxiv.<id>``, so it
    answers for the papers whose only identifier is an arXiv number -- the arXiv API itself
    answers with HTTP 429 when it is rate limited, and a source that cannot be reached may
    not be the only one we ask.
    """

    name = "openalex"
    version = "works-v1"

    def search(self, query: str, *, limit: int = 10) -> SearchResponse:
        response = SearchResponse(
            provider=self.name,
            version=self.version,
            query=query,
            requested_utc=now_utc(),
        )
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"search": query, "per-page": max(int(limit), 1)}
        )
        response.request_url = url
        document, error = _http_json(url)
        if error or not isinstance(document, dict):
            response.error = error or "the response was not an object"
            return response
        for item in document.get("results") or []:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            doi = str(item.get("doi") or "")
            identifier = (doi[len("https://doi.org/"):] if doi.startswith("https://doi.org/")
                          else doi)
            location = item.get("primary_location") or {}
            source = location.get("source") or {}
            venue = str(source.get("display_name") or "") if isinstance(source, dict) else ""
            authors = tuple(
                name for name in (
                    str((entry.get("author") or {}).get("display_name") or "").strip()
                    for entry in (item.get("authorships") or [])
                ) if name
            )
            abstract = _openalex_abstract(item)
            response.records.append(PaperRecord(
                identifier=identifier or str(item.get("id") or ""),
                title=title,
                authors=authors,
                year=int(item["publication_year"]) if item.get("publication_year") else None,
                venue=venue,
                url=str(location.get("landing_page_url") or item.get("id") or ""),
                source=self.name,
                access_level="verified_abstract_only" if abstract else "unavailable",
                retrieved_utc=response.requested_utc,
                abstract=abstract,
                notes="" if abstract else
                "the record carries metadata only; the source returned no abstract",
            ))
        return response


class ArxivProvider:
    """The arXiv Atom API: an identifier, a title, the authors and the abstract."""

    name = "arxiv"
    version = "atom-v1"

    def search(self, query: str, *, limit: int = 10) -> SearchResponse:
        response = SearchResponse(
            provider=self.name,
            version=self.version,
            query=query,
            requested_utc=now_utc(),
        )
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": f"all:{query}", "start": 0, "max_results": max(int(limit), 1)}
        )
        response.request_url = url
        text, error = _http_text(url)
        if error or not text:
            response.error = error or "no response"
            return response
        response.records = _parse_arxiv(text, self.name, response.requested_utc)
        return response


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def _parse_arxiv(text: str, source: str, retrieved: str) -> List[PaperRecord]:
    records: List[PaperRecord] = []
    for entry in re.findall(r"<entry>(.*?)</entry>", text, re.S):
        identifier = _first(entry, "id")
        title = " ".join(_first(entry, "title").split())
        published = _first(entry, "published")
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = tuple(
            " ".join(author.split()) for author in re.findall(r"<name>(.*?)</name>", entry, re.S)
        )
        summary = " ".join(_first(entry, "summary").split())
        records.append(
            PaperRecord(
                identifier=identifier.rsplit("/", 1)[-1] if identifier else "",
                title=title,
                authors=authors,
                year=year,
                venue="arXiv",
                url=identifier,
                source=source,
                access_level="verified_abstract_only",
                retrieved_utc=retrieved,
                abstract=summary,
            )
        )
    return records


def _first(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.S)
    return match.group(1).strip() if match else ""


# ------------------------------------------------------------------ the verification
def verify_record(record: PaperRecord, *, resolver: Optional[Any] = None) -> PaperRecord:
    """Check that a record's identifier resolves and its title matches.

    A record whose identifier does not resolve, or whose title does not match what the
    identifier returns, is a FAILURE for the claim that rests on it -- not a paper with a
    warning attached.
    """
    if not record.identifier:
        record.access_level = "unavailable"
        record.notes = "the record carries no identifier"
        return record
    if resolver is None:
        # no independent resolver was available: the record keeps the level its NAMED
        # provider gave it, and says that the identifier was not resolved independently
        if record.access_level in ("unavailable", ""):
            record.access_level = "verified_abstract_only"
        record.notes = record.notes or (
            "the provider answered with this record; the identifier was not resolved "
            "independently"
        )
        return record
    resolved = resolver(record)
    if resolved is None:
        record.access_level = "unavailable"
        record.notes = "the identifier did not resolve"
        return record
    resolved_title = (resolved.title or "").strip().lower()
    record_title = (record.title or "").strip().lower()
    if resolved_title and record_title and not _titles_agree(resolved_title, record_title):
        record.access_level = "disputed"
        record.notes = (
            f"the title does not match the identifier's: {record.title!r} against "
            f"{resolved.title!r}"
        )
        return record
    if record.access_level == "unavailable":
        record.access_level = "verified_abstract_only"
    record.notes = record.notes or "the identifier resolves and the title matches"
    return record


def _titles_agree(first: str, second: str) -> bool:
    """Whether two titles are the same title, ignoring punctuation and case."""
    def normalise(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    left, right = normalise(first), normalise(second)
    if left == right:
        return True
    return left.startswith(right[:40]) or right.startswith(left[:40])


# ------------------------------------------------------------------ the cache
class LiteratureCache:
    """A cache of searches, keyed by the query, the source, the version and the date."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.entries: Dict[str, Dict[str, Any]] = {}
        if self.path.is_file():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.entries = {}

    @staticmethod
    def key(provider: str, version: str, query: str, day: str) -> str:
        return hashlib.sha256(
            f"{provider}|{version}|{query}|{day}".encode("utf-8")
        ).hexdigest()

    def get(self, provider: str, version: str, query: str) -> Optional[SearchResponse]:
        day = now_utc()[:10]
        entry = self.entries.get(self.key(provider, version, query, day))
        if not entry:
            return None
        response = SearchResponse(
            provider=provider,
            version=version,
            query=query,
            requested_utc=entry.get("requested_utc", now_utc()),
            records=[PaperRecord(**record) for record in entry.get("records") or ()],
            error=entry.get("error", ""),
        )
        return response

    def put(self, response: SearchResponse) -> None:
        day = now_utc()[:10]
        self.entries[self.key(response.provider, response.version, response.query, day)] = {
            "requested_utc": response.requested_utc,
            "records": [record.as_dict() for record in response.records],
            "error": response.error,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.entries, indent=2, default=str) + "\n")
        os.replace(temporary, self.path)


# ------------------------------------------------------------------ the agent
class LiteratureAgent:
    """The literature agent: it asks the providers and it records what it asked."""

    def __init__(
        self,
        out_dir: Path = LITERATURE_DIR,
        providers: Optional[Sequence[Provider]] = None,
        resolver: Optional[Any] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.providers = list(providers) if providers is not None else [
            CrossrefProvider(),
            OpenAlexProvider(),
            ArxivProvider(),
        ]
        self.resolver = resolver
        self.cache = LiteratureCache(self.out_dir / "cache.json")

    def search(
        self, query: str, *, limit: int = 10, verify: bool = True
    ) -> Dict[str, Any]:
        """Search every provider, verify the records, and record the requests."""
        responses: List[SearchResponse] = []
        papers: Dict[str, PaperRecord] = {}
        for provider in self.providers:
            cached = self.cache.get(provider.name, provider.version, query)
            response = cached if cached is not None else provider.search(query, limit=limit)
            if cached is None:
                self.cache.put(response)
            responses.append(response)
            for record in response.records:
                if verify:
                    record = verify_record(record, resolver=self.resolver)
                papers.setdefault(record.identity, record)
        reachable = [r for r in responses if not r.error]
        if not reachable:
            self._write(
                search_log=responses,
                papers=list(papers.values()),
                review={
                    "state": LITERATURE_BLOCKED,
                    "query": query,
                    "reason": "no external source answered: "
                    + "; ".join(f"{r.provider}: {r.error}" for r in responses),
                    "operator_action": "configure an academic provider, or supply a verified "
                    "cache, before any novelty claim is made",
                },
                claims=[],
            )
            raise LiteratureBlocked(
                f"no external source answered the query {query!r}; the novelty claim is blocked"
            )
        review = {
            "state": "searched",
            "query": query,
            "providers": [
                {"name": r.provider, "version": r.version, "count": len(r.records), "error": r.error}
                for r in responses
            ],
            "papers": len(papers),
        }
        self._write(search_log=responses, papers=list(papers.values()), review=review, claims=[])
        return review

    def record_claim(
        self,
        claim: str,
        *,
        paper_identifiers: Sequence[str],
        section: str = "",
        status: str = "verified_abstract_only",
        note: str = "",
    ) -> Dict[str, Any]:
        """Record a novelty claim and check that its papers are verified.

        A claim that cites an unverified paper, or that has no paper at all, is a FAILURE:
        the claim is recorded as failed rather than quietly written down.
        """
        papers = self._papers()
        by_identity = {paper.identity: paper for paper in papers}
        cited: List[PaperRecord] = []
        missing: List[str] = []
        for identifier in paper_identifiers:
            record = by_identity.get((identifier or "").strip().lower())
            if record is None:
                missing.append(identifier)
            else:
                cited.append(record)
        failed = bool(missing) or not cited or any(
            paper.access_level not in ("verified_fulltext", "verified_abstract_only")
            for paper in cited
        )
        claim_record = {
            "claim": claim,
            "papers": [paper.identifier for paper in cited],
            "missing_papers": missing,
            "section": section,
            "evidence_status": "failed" if failed else status,
            "note": note,
            "recorded_utc": now_utc(),
        }
        claims = self._claims()
        claims.append(claim_record)
        self._write_claims(claims)
        return claim_record

    # ------------------------------------------------------------------ the files
    def _papers(self) -> List[PaperRecord]:
        path = self.out_dir / "papers.jsonl"
        if not path.is_file():
            return []
        records: List[PaperRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(PaperRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        return records

    def _claims(self) -> List[Dict[str, Any]]:
        path = self.out_dir / "novelty_claims.json"
        if not path.is_file():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("claims") or []
        except (OSError, json.JSONDecodeError):
            return []

    def _write_claims(self, claims: List[Dict[str, Any]]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "novelty_claims.json").write_text(
            json.dumps({"claims": claims}, indent=2, default=str) + "\n", encoding="utf-8"
        )

    def _write(
        self,
        *,
        search_log: Sequence[SearchResponse],
        papers: Sequence[PaperRecord],
        review: Dict[str, Any],
        claims: Sequence[Dict[str, Any]],
    ) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / "search_log.jsonl").open("a", encoding="utf-8") as handle:
            for response in search_log:
                handle.write(json.dumps(response.as_dict(), sort_keys=True) + "\n")
        # The paper store ACCUMULATES. A store that only kept the last query's papers would
        # silently drop the identities an earlier query verified, and a literature registry
        # that forgets its own verified papers is not a registry.
        known: Dict[str, Dict[str, Any]] = {}
        papers_path = self.out_dir / "papers.jsonl"
        if papers_path.is_file():
            for line in papers_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    stored = json.loads(line)
                except json.JSONDecodeError:
                    continue
                identity = str(stored.get("identifier") or stored.get("title") or "").strip().lower()
                if identity:
                    known[identity] = stored
        for paper in papers:
            known[paper.identity] = paper.as_dict()
        with papers_path.open("w", encoding="utf-8") as handle:
            for stored in sorted(known.values(),
                                key=lambda item: (str(item.get("identifier") or ""),
                                                  str(item.get("title") or ""))):
                handle.write(json.dumps(stored, sort_keys=True, default=str) + "\n")
        (self.out_dir / "literature_review.json").write_text(
            json.dumps(review, indent=2, default=str) + "\n", encoding="utf-8"
        )
        if claims:
            self._write_claims(list(claims))


__all__ = [
    "ArxivProvider",
    "CrossrefProvider",
    "OpenAlexProvider",
    "EVIDENCE_LEVELS",
    "LITERATURE_BLOCKED",
    "LITERATURE_DIR",
    "LiteratureAgent",
    "LiteratureBlocked",
    "LiteratureCache",
    "PaperRecord",
    "Provider",
    "SearchResponse",
    "verify_record",
]
