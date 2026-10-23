"""Agent L's tests (issue #110, WP1).

The literature agent's whole job is to not make things up. The tests therefore cover the
failure modes: a provider that answers nothing blocks the novelty claim, a record whose
identifier does not resolve is unavailable, a record whose title disagrees with its
identifier is disputed, and a claim that cites an unverified paper fails. A search is
recorded as a request and a response, never as a claim that a search happened.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/harness2_genrec/tests/test_literature.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.harness2_genrec import literature as L  # noqa: E402


class _Provider:
    name = "fake"
    version = "v1"

    def __init__(self, records=None, error=""):
        self.records = records or []
        self.error = error
        self.asked = []

    def search(self, query, *, limit=10):
        self.asked.append(query)
        response = L.SearchResponse(
            provider=self.name,
            version=self.version,
            query=query,
            requested_utc=L.now_utc(),
            error=self.error,
        )
        response.records = list(self.records)
        return response


def _record(identifier="10.1000/abc", title="Hyperbolic semantic IDs", level="verified_abstract_only"):
    return L.PaperRecord(
        identifier=identifier,
        title=title,
        authors=("A. Author",),
        year=2024,
        url=f"https://doi.org/{identifier}",
        source="fake",
        access_level=level,
        retrieved_utc=L.now_utc(),
    )


def test_a_search_records_the_request_and_the_response(tmp_path):
    provider = _Provider([_record()])
    agent = L.LiteratureAgent(out_dir=tmp_path, providers=[provider])
    review = agent.search("hyperbolic semantic id")
    assert provider.asked == ["hyperbolic semantic id"]
    assert review["state"] == "searched"
    log = (tmp_path / "search_log.jsonl").read_text().strip().splitlines()
    assert json.loads(log[0])["query"] == "hyperbolic semantic id"
    assert json.loads(log[0])["count"] == 1
    papers = (tmp_path / "papers.jsonl").read_text().strip().splitlines()
    assert json.loads(papers[0])["identifier"] == "10.1000/abc"


def test_a_provider_that_answers_nothing_blocks_the_claim(tmp_path):
    provider = _Provider([], error="HTTP 503")
    agent = L.LiteratureAgent(out_dir=tmp_path, providers=[provider])
    with pytest.raises(L.LiteratureBlocked):
        agent.search("hyperbolic quantisation")
    review = json.loads((tmp_path / "literature_review.json").read_text())
    assert review["state"] == L.LITERATURE_BLOCKED
    assert "operator_action" in review


def test_an_identifier_that_does_not_resolve_is_unavailable():
    record = L.verify_record(_record(), resolver=lambda record: None)
    assert record.access_level == "unavailable"
    assert "did not resolve" in record.notes


def test_a_title_that_disagrees_with_the_identifier_is_disputed():
    resolved = _record(title="A different paper entirely")
    record = L.verify_record(_record(), resolver=lambda record: resolved)
    assert record.access_level == "disputed"
    assert "does not match" in record.notes


def test_a_resolving_identifier_with_a_matching_title_is_verified():
    resolved = _record()
    record = L.verify_record(_record(), resolver=lambda record: resolved)
    assert record.access_level == "verified_abstract_only"
    assert "resolves" in record.notes


def test_a_claim_that_cites_an_unverified_paper_fails(tmp_path):
    provider = _Provider([_record(level="unavailable")])
    agent = L.LiteratureAgent(out_dir=tmp_path, providers=[provider], resolver=lambda record: None)
    agent.search("hyperbolic")
    claim = agent.record_claim(
        "a hyperbolic quantiser is novel here", paper_identifiers=["10.1000/abc"]
    )
    assert claim["evidence_status"] == "failed"


def test_a_claim_with_a_verified_paper_is_recorded(tmp_path):
    provider = _Provider([_record()])
    agent = L.LiteratureAgent(
        out_dir=tmp_path, providers=[provider], resolver=lambda record: _record()
    )
    agent.search("hyperbolic")
    claim = agent.record_claim(
        "a hyperbolic quantiser is novel here",
        paper_identifiers=["10.1000/abc"],
        section="3.2",
        status="verified_abstract_only",
    )
    assert claim["evidence_status"] == "verified_abstract_only"
    assert claim["section"] == "3.2"
    claims = json.loads((tmp_path / "novelty_claims.json").read_text())["claims"]
    assert claims[-1]["papers"] == ["10.1000/abc"]


def test_a_claim_that_cites_nothing_fails(tmp_path):
    provider = _Provider([_record()])
    agent = L.LiteratureAgent(
        out_dir=tmp_path, providers=[provider], resolver=lambda record: _record()
    )
    agent.search("hyperbolic")
    claim = agent.record_claim("an unsupported claim", paper_identifiers=[])
    assert claim["evidence_status"] == "failed"


def test_the_cache_keeps_a_search_from_being_repeated(tmp_path):
    provider = _Provider([_record()])
    agent = L.LiteratureAgent(out_dir=tmp_path, providers=[provider])
    agent.search("hyperbolic")
    agent.search("hyperbolic")
    assert provider.asked == ["hyperbolic"]
