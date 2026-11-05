# The prompt template of the `ideate` agent

Source: `tools/research/rounds/step2_ideation.py` -> `ideate_one()`

```python
def ideate_one(root: Path, objective: str, extra: bool, index: int) -> tuple:
    """One ideation agent: its own task id, prompt, workspace and answer file.

    The function is the unit the round runs concurrently and the unit the
    flow dispatches it concurrently; every caller uses it, so the workflows
    cannot drift apart.
    """
    prompt_base = agent_contracts.IDEA_PROMPT.format(
        minimum=IDEA_MIN, maximum=IDEA_MAX,
        agents=pipeline_module.IDEA_AGENTS,
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
        agent_contracts.EXPERIMENT_POLICY)
    task_id = "ideation_" + str(index)
    prompt = (prompt_base + "\n\nThe round objective: " + (objective or "improve the "
                 "item semantic IDs for a generative recommender") +
                 "\nYour answer must be a different set of ideas from the other " +
                 str(pipeline_module.IDEA_AGENTS - 1) +
                 " agents, which you cannot see.")
    history = round_history(Path(root))
    if history:
        prompt += ("\n\nWHAT THE EARLIER ROUNDS MEASURED, AND WHAT THEY FAILED ON:\n"
                   + history +
                   "\nUse it to aim at something the earlier rounds did not measure. "
                   "Do not restate a proposal that was already made and rejected.")
    if extra:
        prompt += ("\n\nThe first set of ideas you gave has been collected. Give one "
                   "MORE idea, and it must be different from the one you gave before "
                   "and from what the other agents propose.")
    return (task_id, _run_agent(root, task_id, prompt,
                                "ideas_extra.txt" if extra else "ideas.txt",
                                plaintext.parse_ideas,
                                role="ideate", read_paths=[],
                                allowed_roots=_round_roots(root)))
```
