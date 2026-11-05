# The prompt template of the `review_math` agent

Source: `tools/research/rounds/worker.py` -> `_review_math_prompt()`

```python
def _review_math_prompt(idea: dict, spec: dict) -> str:
    """The prompt of a Math Reviewer: the idea and the specification."""
    return (
        "Check this mathematical specification against the idea it came from.\n\n"
        "IDEA TITLE: " + str(idea.get("title")) + "\n"
        "IDEA SPECIFICATION: " + str(idea.get("specification")) + "\n"
        "IDEA SPEC EQUATIONS: " + str(idea.get("spec_equations")) + "\n"
        "IDEA COUPLING: " + str(idea.get("coupling_path")) + "\n"
        "IDEA ABLATIONS: " + str(idea.get("ablation_behavior_off")) + " | "
        + str(idea.get("ablation_geometry_off")) + "\n\n"
        "THE SPECIFICATION:\n" + json.dumps(spec, indent=1, ensure_ascii=False)
        + "\n\nANSWER CONTRACT. Answer with one JSON object:\n"
        '{"verdict": "PASS" | "REVISE", "findings": ["..."], '
        '"tests": [{"finding": "...", "eval_id": "...", "code": "..."}], '
        '"reason": "..."}\n'
        "Every finding MUST carry a test. A test is a Python module that defines "
        "check(spec) -> list[str]: it returns a non-empty list of problems exactly "
        "when the specification shows the defect you describe, and an empty list "
        "otherwise. It may only read the specification dictionary; it must not "
        "import torch or the agents SDK, run training, or read data. The framework "
        "runs your test against the specification you are reviewing: a test that "
        "does not fail on it is refused as not being a test of that defect. Your "
        "tests are written to tools/research/rounds/spec_tests/ and are run on "
        "every later round before it may freeze, so write them to refuse the "
        "defect, not to pass.")
```
