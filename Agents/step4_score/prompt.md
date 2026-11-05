# The prompt template of the `score` agent

Source: `tools/research/rounds/step4_scoring.py` -> `_rank_with_retry()`

```python
def _rank_with_retry(root: Path, task_id: str, prompt: str) -> dict:
    """One ranking, and one bounded retry if its schema is broken.

    The ranking schema is strict on purpose: a missing proposal, a repeated one
    or a tied position is an error, not a partial average. A reviewer that
    breaks it is told what was wrong and asked once more, with a fresh
    workspace, so a stale answer can never be reused.
    """
    attempt = 0
    last_error = None
    # The budget belongs to the task, not to the attempt: a retry does not get
    # a second five minutes.
    import time as _time
    budget = float(settings.AGENT_DEADLINE_SECONDS)
    started = _time.monotonic()
    while attempt < 2:
        remaining = budget - (_time.monotonic() - started)
        if remaining <= 1.0:
            break
        suffix = "" if attempt == 0 else (
            "\n\nYour previous answer was rejected by the schema: " + str(last_error) +
            "\nAnswer again with one complete block per proposal, and keep the five lines "
            "of a block together:\n"
            "SCORE: <idea_id> | behavior_evidence=<0-5> | geometry_necessity=<0-5> | "
            "novelty=<0-5> | downstream_mechanism=<0-5> | falsifiability=<0-5> | "
            "implementability=<0-5> | exploration_value=<0-5>\n"
            "SCOPE: <idea_id> | IN_SCOPE or OUT_OF_SCOPE | <why>\n"
            "EVIDENCE: <idea_id> | <evidence ids>\n"
            "OBJECTION: <idea_id> | <the strongest objection>\n"
            "RATIONALE: <idea_id> | <why these numbers, and the uncertainty>\n"
            "Every dimension value is a number from 0 to 5. A dimension you cannot ground "
            "takes the number you can defend, and the RATIONALE says the dimension is "
            "UNKNOWN; UNKNOWN is never a dimension value. Every proposal you score needs "
            "its own RATIONALE line, carrying its own idea_id.")
        name = "scores.txt" if attempt == 0 else "scores_retry.txt"
        run_id = task_id if attempt == 0 else task_id + "_retry1"
        try:
            report = _run_agent(root, run_id, prompt + suffix, name,
                                plaintext.parse_ranks, role="score",
                                read_paths=[],
                                allowed_roots=_round_roots(root),
                                deadline_seconds=max(1, int(remaining)))
            scoring_module.validate_report(report, reviewer=task_id)
            return report
        except (scoring_module.ScoreError, pipeline_module.RoundError) as exc:
            last_error = exc
            attempt += 1
    raise pipeline_module.RoundError("reviewer " + task_id + " broke the ranking schema "
                                     "twice: " + str(last_error))
```
