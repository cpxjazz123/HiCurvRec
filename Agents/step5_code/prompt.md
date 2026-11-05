# The prompt template of the `code` agent

Source: `tools/research/rounds/worker.py` -> `_coding_prompt()`

```python
def _coding_prompt(spec: dict) -> str:
    """The prompt of a Coding Agent: the frozen specification, and nothing else."""
    return (
        "Implement this frozen mathematical specification. Do not reinterpret "
        "it: every sign, scale, limit and switch is as written.\n\n"
        + json.dumps(spec, indent=1, ensure_ascii=False))
```
