# The literature source policy (issue #110, sections 1.A and 5)

What counts as a verified source, how a search is recorded, and what happens when no source
is reachable. The rule is short: **a fabricated or non-resolving citation is a hard failure
for that claim**, and when no external search is available the subsystem persists
`LITERATURE_BLOCKED` rather than inventing novelty.

## The evidence levels

| Status | What it means |
|---|---|
| `verified_fulltext` | the primary source was actually read, and the method content was taken from it |
| `verified_abstract_only` | the metadata and the abstract were verified, and nothing more is claimed |
| `disputed` | the sources conflict, and the conflict is reported rather than resolved silently |
| `unavailable` | the source could not be reached |

Only `verified_fulltext` supports an extracted equation. The other levels support a statement
about the paper's existence, title, authors, year and identifier — nothing more.

## What a paper record must carry

The canonical title, the authors, the year, the DOI or the arXiv identifier, the URL, the
venue, the access level, the retrieval date, and the section or page the claim came from.
A title that does not match its identifier, or an identifier that does not resolve, is a
failure of the record, not a detail.

## The output files

| File | What it holds |
|---|---|
| `search_log.jsonl` | every source-query request and response, with the date, so a search is never merely claimed |
| `papers.jsonl` | the deduplicated paper identities with their provenance |
| `literature_review.json` | the review: the methods, the equations or the precise paraphrases, the geometry, the curvature, the metrics, the operators, the optimiser and the limits |
| `novelty_claims.json` | each claim, with the paper identifiers it rests on, the access level, the supporting section, and its evidence status |

## The cache

A search is cached by the query, the source, the provider version and the date, so the same
question is not asked twice and a cached record keeps its retrieval date. A cached record may
be reused for a novelty claim; it may not be presented as a new search.

## The prior art

A mathematical idea is not novel until the repository's own history has been read: the R8
arms (including the four hyperbolic ones, whose mathematics is recoverable from their
source), the 38-model inventory, and the previous campaigns' proposals in every version. A
mechanism whose mathematics cannot be recovered from the archives is `novelty_unknown`, and
it blocks the cold launch until it is inspected. A name is never evidence of novelty.

## When nothing is reachable

The subsystem persists `LITERATURE_BLOCKED` with the operator action it needs — configure a
provider, or supply a verified cache — and the campaign does not dispatch a new novelty
claim. A round with no new evidence describes the blocker; it does not conjure a discovery.
