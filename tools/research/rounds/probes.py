"""Bounded, train-only evidence probes (Issue #123).

Before the reviewers see a proposal, the framework measures what the proposal
claims: the train-only behaviour statistics behind its hypothesis, the sanity of
its curved operation on small tensors, and (separately) whether the related work
can be traced. Everything here is read-only, CPU-only, deterministic and capped:
a probe never trains, never edits a tree, and never touches validation or test
data.

What is *not* available is reported as such. ``DATA_UNAVAILABLE`` means the
train split does not carry the field the hypothesis needs, ``UNKNOWN`` means the
probe could not establish it, and ``NOT_IMPLEMENTED`` means this probe has no
diagnostic for that family of hypothesis. None of them is a measurement, and
none of them is evidence that the idea is wrong.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

#: The frozen data contract. The probes read the train split and the on-disk
#: caches; they never read validation, test or the Stage 3 labels.
TRAIN_PARQUET = "results/stage0_build_parquet/train.parquet"
ITEMS_PARQUET = "results/stage0_build_parquet/items.parquet"
EMBEDDINGS = "stage1_GeneEmbedding/output/sentence_t5.npy"
BASELINE_SID = "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json"

#: The initial resource envelope. These are policy limits the orchestrator
#: enforces, not promises about how long a probe takes.
DEFAULT_BUDGET_SECONDS = 120.0
DEFAULT_MAX_USERS = 20000
DEFAULT_SEED = 20261025
MAX_SAMPLE_ROWS = 400000

STATUS_OK = "OK"
STATUS_DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"
STATUS_NOT_RUN = "NOT_RUN"
STATUS_PARTIAL = "PARTIAL"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip(value, low, high):
    return max(low, min(high, value))


def discover_schema(repo: Path) -> dict:
    """What the train split actually carries.

    The column names are read from the file, not assumed: a hypothesis that
    needs ordered user histories has to be able to say whether they are there.
    """
    import pandas as pd

    train = Path(repo) / TRAIN_PARQUET
    if not train.is_file():
        return {"status": STATUS_DATA_UNAVAILABLE,
                "reason": "the train parquet is not on disk",
                "source": TRAIN_PARQUET}
    frame = pd.read_parquet(train)
    columns = {str(name): str(dtype) for name, dtype in frame.dtypes.items()}
    has_user = "user" in frame.columns
    has_history = "history" in frame.columns or "seen_history" in frame.columns
    has_target = "target" in frame.columns
    users = int(frame["user"].nunique()) if has_user else None
    items = set()
    for column in ("history", "seen_history"):
        if column in frame.columns:
            for value in frame[column].head(5000):
                items.update(int(item) for item in list(value))
    if has_target:
        items.update(int(value) for value in frame["target"].head(200000))
    return {
        "status": STATUS_OK,
        "source": TRAIN_PARQUET,
        "source_sha256": _sha256(train),
        "rows": int(len(frame)),
        "columns": columns,
        "unique_users": users,
        "has_user": has_user,
        "has_ordered_history": has_history,
        "has_target": has_target,
        "distinct_items_seen": len(items),
        "leakage_note": "the train split only; validation, test and the Stage 3 labels "
                        "are never read here",
    }


def _sequences(frame, max_users: int) -> list:
    """The per-user ordered item sequences, read from the train split only."""
    import pandas as pd

    columns = list(frame.columns)
    history_column = "seen_history" if "seen_history" in columns else (
        "history" if "history" in columns else None)
    if "user" not in columns or history_column is None:
        return []
    frame = frame.sort_values("user", kind="stable")
    sequences = []
    for user, group in frame.groupby("user", sort=False):
        if len(sequences) >= max_users:
            break
        last = group.iloc[-1]
        sequence = [int(item) for item in list(last[history_column])]
        if "target" in columns and pd.notna(last.get("target")):
            sequence.append(int(last["target"]))
        if len(sequence) >= 2:
            sequences.append(sequence)
    return sequences


def transition_metrics(sequences: list, *, seed: int = DEFAULT_SEED) -> dict:
    """Train-only directed transition statistics of the user's sequences."""
    import random

    edges = {}
    outgoing = {}
    incoming = {}
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            edges[(left, right)] = edges.get((left, right), 0) + 1
            outgoing[left] = outgoing.get(left, 0) + 1
            incoming[right] = incoming.get(right, 0) + 1
    sources = len(outgoing)
    targets = len(incoming)
    total = sum(edges.values())
    if not total:
        return {"status": STATUS_DATA_UNAVAILABLE,
                "reason": "no ordered user sequence yields a single transition edge"}
    freqs = {}
    for (left, right), count in edges.items():
        freqs.setdefault(left, {})[right] = count
    entropies = []
    for left, row in freqs.items():
        mass = sum(row.values())
        entropies.append(-sum((count / mass) * math.log(count / mass)
                              for count in row.values()))
    popularity = {}
    for (left, right), count in edges.items():
        popularity[right] = popularity.get(right, 0) + count
    best_popular = max(popularity.values()) if popularity else 0
    # How often the single most popular successor is the observed one: a cheap
    # reference for "is the transition signal more than popularity".
    hits = 0
    for left, row in freqs.items():
        top = max(popularity.items(), key=lambda item: item[1])[0]
        hits += row.get(top, 0)
    counts = sorted(outgoing.values())
    quantiles = {
        "p10": counts[int(0.1 * (len(counts) - 1))] if counts else 0,
        "median": counts[len(counts) // 2] if counts else 0,
        "p90": counts[int(0.9 * (len(counts) - 1))] if counts else 0,
    }
    # A seeded permutation of the sequences gives a train-only null for the
    # "is this more than chance" question.
    rng = random.Random(seed)
    shuffled = []
    for sequence in sequences:
        copy = list(sequence)
        rng.shuffle(copy)
        shuffled.append(copy)
    null_edges = 0
    for sequence in shuffled:
        null_edges += len(sequence) - 1
    return {
        "status": STATUS_OK,
        "edges": total,
        "distinct_edges": len(edges),
        "source_items": sources,
        "target_items": targets,
        "edge_density": round(len(edges) / max(1, sources), 4),
        "mean_outgoing_degree": round(total / max(1, sources), 4),
        "outgoing_degree_quantiles": quantiles,
        "mean_transition_entropy_nats": round(sum(entropies) / max(1, len(entropies)), 4),
        "popular_successor_hit_rate": round(hits / total, 4),
        "popularity_only_edge_mass": round(best_popular / total, 4),
        "permuted_null_edges": null_edges,
        "coverage_note": "counts are over the sampled users only; the coverage is the "
                         "number of users and items the probe read",
    }


def branch_metrics(sequences: list, *, seed: int = DEFAULT_SEED) -> dict:
    """Train-only branching structure of the items."""
    import random

    successors = {}
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            successors.setdefault(left, set()).add(right)
    if not successors:
        return {"status": STATUS_DATA_UNAVAILABLE,
                "reason": "no item has an observed successor"}
    degrees = sorted(len(value) for value in successors.values())
    rng = random.Random(seed)
    null_degrees = []
    flat = [item for sequence in sequences for item in sequence]
    for _ in range(min(len(sequences), 2000)):
        copy = list(flat)
        rng.shuffle(copy)
        window = len(sequences[0]) if sequences else 2
        for start in range(0, min(len(copy), window * 50), max(1, window)):
            chunk = copy[start:start + window]
            for left, right in zip(chunk, chunk[1:]):
                null_degrees.append(1)
    entropy = []
    for left, values in successors.items():
        size = len(values)
        entropy.append(math.log(size) if size > 1 else 0.0)
    return {
        "status": STATUS_OK,
        "items_with_successors": len(successors),
        "branch_degree_quantiles": {
            "p10": degrees[int(0.1 * (len(degrees) - 1))] if degrees else 0,
            "median": degrees[len(degrees) // 2] if degrees else 0,
            "p90": degrees[int(0.9 * (len(degrees) - 1))] if degrees else 0,
        },
        "mean_branch_entropy_nats": round(sum(entropy) / max(1, len(entropy)), 4),
        "coverage_note": "branching is measured on the train split only",
    }


def prefix_metrics(sequences: list, repo: Path, *, seed: int = DEFAULT_SEED) -> dict:
    """Do the baseline SID prefixes keep train-only transition groups apart?

    This is a diagnostic about the *existing* baseline SIDs. It is never a
    quality floor: a number here cannot refuse a candidate.
    """
    import random

    sid_path = Path(repo) / BASELINE_SID
    if not sid_path.is_file():
        return {"status": STATUS_DATA_UNAVAILABLE,
                "reason": "the baseline SID export is not on disk",
                "source": BASELINE_SID}
    try:
        codes = json.loads(sid_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": STATUS_UNKNOWN, "reason": "the baseline SID file is unreadable: " + str(exc)}
    if not isinstance(codes, dict):
        return {"status": STATUS_UNKNOWN, "reason": "the baseline SID file has an unknown shape"}
    prefix_of = {}
    for key, value in codes.items():
        try:
            identifier = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, (list, tuple)) and value:
            prefix_of[identifier] = tuple(int(token) for token in value[:3])
    total = kept = 0
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            if left not in prefix_of or right not in prefix_of:
                continue
            total += 1
            if prefix_of[left][:1] == prefix_of[right][:1]:
                kept += 1
    if not total:
        return {"status": STATUS_DATA_UNAVAILABLE,
                "reason": "no train transition has both endpoints in the baseline SID map",
                "source": BASELINE_SID, "source_sha256": _sha256(sid_path)}
    rng = random.Random(seed)
    shuffled = []
    for sequence in sequences:
        copy = list(sequence)
        rng.shuffle(copy)
        shuffled.append(copy)
    null_total = null_kept = 0
    for sequence in shuffled:
        for left, right in zip(sequence, sequence[1:]):
            if left not in prefix_of or right not in prefix_of:
                continue
            null_total += 1
            if prefix_of[left][:1] == prefix_of[right][:1]:
                null_kept += 1
    return {
        "status": STATUS_OK,
        "source": BASELINE_SID,
        "source_sha256": _sha256(sid_path),
        "train_transitions_with_both_endpoints": total,
        "same_first_prefix_rate": round(kept / total, 4),
        "permuted_null_same_first_prefix_rate": (round(null_kept / null_total, 4)
                                                if null_total else None),
        "note": "a diagnostic about the existing baseline SIDs; never an admission gate",
    }


def context_metrics(sequences: list) -> dict:
    """How many distinct training contexts each item sees."""
    contexts = {}
    for sequence in sequences:
        for index, item in enumerate(sequence):
            prefix = tuple(sequence[:index])
            contexts.setdefault(item, set()).add(prefix)
    if not contexts:
        return {"status": STATUS_DATA_UNAVAILABLE,
                "reason": "no item appears in a training context"}
    sizes = sorted(len(value) for value in contexts.values())
    return {
        "status": STATUS_OK,
        "items_with_contexts": len(contexts),
        "contexts_per_item_quantiles": {
            "p10": sizes[int(0.1 * (len(sizes) - 1))] if sizes else 0,
            "median": sizes[len(sizes) // 2] if sizes else 0,
            "p90": sizes[int(0.9 * (len(sizes) - 1))] if sizes else 0,
        },
        "coverage_note": "contexts are the prefixes of the train sequences only",
    }


def hypothesis_family(idea: dict) -> str:
    """Which family of hypothesis the proposal states."""
    text = " ".join(str(idea.get(field, "")) for field in
                     ("title", "key_change", "behavior_signal", "coupling_path",
                      "observables")).lower()
    if "prefix" in text or "cluster" in text or "group" in text:
        return "prefix"
    if "branch" in text or "divers" in text or "entropy" in text:
        return "branch"
    if "transition" in text or "successor" in text or "next-item" in text \
            or "next item" in text:
        return "transition"
    if "cohort" in text or "context" in text:
        return "cohort"
    return "other"


def run_probe(idea: dict, *, repo: Path, budget_seconds: float = DEFAULT_BUDGET_SECONDS,
              max_users: int = DEFAULT_MAX_USERS,
              seed: int = DEFAULT_SEED) -> dict:
    """Run the probe of one proposal and return its report."""
    import pandas as pd

    started = time.monotonic()
    repo = Path(repo)
    report = {
        "schema": 1,
        "idea_id": idea.get("idea_id"),
        "hypothesis_family": hypothesis_family(idea),
        "budget_seconds": budget_seconds,
        "seed": seed,
        "started_at": started,
    }
    schema = discover_schema(repo)
    report["schema_discovery"] = schema
    if schema.get("status") != STATUS_OK:
        report["status"] = STATUS_DATA_UNAVAILABLE
        report["reason"] = schema.get("reason")
        report["metrics"] = {}
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return report
    frame = pd.read_parquet(Path(repo) / TRAIN_PARQUET)
    if len(frame) > MAX_SAMPLE_ROWS:
        frame = frame.sample(n=MAX_SAMPLE_ROWS, random_state=seed).sort_index()
        report["sampled"] = True
        report["sampled_rows"] = int(len(frame))
    sequences = _sequences(frame, max_users)
    report["sequences_read"] = len(sequences)
    report["users_read"] = len(sequences)
    if not sequences:
        report["status"] = STATUS_DATA_UNAVAILABLE
        report["reason"] = "the train split carries no ordered user sequence"
        report["metrics"] = {}
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return report
    family = report["hypothesis_family"]
    metrics = {}
    try:
        if family in ("transition", "prefix"):
            metrics["transition"] = transition_metrics(sequences, seed=seed)
            if family == "prefix":
                metrics["prefix"] = prefix_metrics(sequences, repo, seed=seed)
        elif family == "branch":
            metrics["branch"] = branch_metrics(sequences, seed=seed)
            metrics["transition"] = transition_metrics(sequences, seed=seed)
        elif family == "cohort":
            metrics["cohort"] = context_metrics(sequences)
            metrics["transition"] = transition_metrics(sequences, seed=seed)
        else:
            metrics["note"] = {
                "status": STATUS_NOT_IMPLEMENTED,
                "reason": "this probe has no diagnostic for the hypothesis family of "
                          "this proposal; the proposal must name the statistic it needs",
            }
    except Exception as exc:                              # noqa: BLE001
        metrics = {"error": {"status": STATUS_UNKNOWN, "reason": str(exc)[:200]}}
    report["metrics"] = metrics
    usable = [value for value in metrics.values()
              if isinstance(value, dict) and value.get("status") == STATUS_OK]
    report["status"] = STATUS_OK if usable else STATUS_PARTIAL
    if not usable:
        report["reason"] = "no diagnostic could be measured for this hypothesis"
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["within_budget"] = report["elapsed_seconds"] <= budget_seconds
    return report


def _cli() -> int:                                    # pragma: no cover - entry point
    """Run one probe and print its report as JSON.

    The probe needs the project's interpreter (pandas and pyarrow); the MCP
    server's own interpreter does not carry them, so the orchestrator runs this
    module as a bounded subprocess instead of importing it in place.
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="train-only evidence probe")
    parser.add_argument("--idea", required=True, help="the proposal as JSON")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--max-users", type=int, default=DEFAULT_MAX_USERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    idea = json.loads(arguments.idea)
    report = run_probe(idea, repo=Path(arguments.repo),
                       budget_seconds=arguments.budget,
                       max_users=arguments.max_users, seed=arguments.seed)
    print(json.dumps(report, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(_cli())
