# The prompt template of the `start` agent

Source: `tools/research/rounds/pipeline.py` -> `new_round()`

```python
def new_round(root: Path, round_id: str, *, objective: str = "",
              issue: str = DEFAULT_ISSUE) -> Path:
    """Create the round directory and its initial state."""
    directory = round_root(root, issue) / round_id
    if (directory / "state.json").is_file():
        raise RoundError("round already exists: " + round_id)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": 1,
        "round_id": round_id,
        "issue": issue,
        "objective": objective,
        "created_at": time.time(),
        "step": "ideation",
        "probe_state": PROBE_PENDING,
        "ideas": [],
        "ideas_rejected": [],
        "duplicates": [],
        "scores": {},
        "matrix": None,
        "selection": None,
        "implementations": [],
        "blocked": [],
        "stage2_jobs": [],
        "stage3_jobs": [],
        "events": [{"event": "round_created", "at": time.time()}],
    }
    atomic_json(directory / "state.json", state)
    return directory
```
