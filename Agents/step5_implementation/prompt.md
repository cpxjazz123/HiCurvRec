# The prompt template of the `implementation` agent

Source: `tools/research/rounds/step6_implementation.py` -> `run_implementation()`

```python
def run_implementation(root: Path, idea: dict, tree: Path) -> dict:
    """One implementation agent per selected idea, in its own tree.

    The tree is a fresh copy of the one permitted baseline, made here, so the
    agent never starts from another candidate's work and never from a mutated
    worktree.
    """
    tree = Path(tree)
    provenance = baseline_module.copy_baseline(tree)
    prompt = (agent_contracts.IMPLEMENT_PROMPT.format(
        idea=plaintext.render_idea(idea), tree=str(tree),
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
        agent_contracts.EXPERIMENT_POLICY))
    # No implementation agent runs in the Issue #121 flow: the framework copies
    # the baseline and the main agent writes the mechanism. If this role is
    # re-enabled, it also needs the Stage 0/1 inputs and the interpreter as read
    # paths (it must run the smoke); the tree alone is not enough.
    report = _run_agent(root, "implementation_" + idea["idea_id"], prompt,
                        "implementation.txt", plaintext.parse_implementation,
                        role="implementation")
    report["provenance"] = provenance
    report["baseline"] = provenance["baseline"]
    report["baseline_digest"] = provenance["baseline_digest"]
    report["candidate_digest_before_agent"] = provenance["candidate_digest"]
    atomic_json(tree / "PROPOSAL.json", idea)
    # Issue #124: the candidate's own path is checked, not just the generic
    # probe. A candidate whose implemented operator is insensitive to the
    # behaviour signal, or that loses the gradient, is blocked here.
    try:
        from . import candidate_checks
        check = candidate_checks.run_candidate_check(tree, repo=_repository_root(Path(root)))
    except Exception as exc:                              # noqa: BLE001
        check = {"verdict": "INCONCLUSIVE_OR_INFRA_ERROR",
                 "reason": "the candidate path check crashed: " + type(exc).__name__}
    report["candidate_path_check"] = check
    report["candidate_digest_after_agent"] = provenance["candidate_digest"]
    atomic_json(tree / "CANDIDATE_CHECK.json", check)
    return report



#: The cap on the diff a reviewer is shown. A revision that rewrites half the
#: tree is reviewed from the whole file list anyway; the patch names what moved.
DIFF_LIMIT = 60000

#: Files that belong to a run, not to the source: never part of a snapshot or a
#: diff, because they say nothing about what was implemented.
_DIFF_IGNORED = ("logs", "results", "dataset", "__pycache__", "ckpt", ".git")
```
