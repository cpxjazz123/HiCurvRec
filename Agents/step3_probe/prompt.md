# The prompt template of the `probe` agent

Source: `tools/research/rounds/probes.py` -> `run_probe()`

```python
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
```
