# The prompt template of the `verify` agent

Source: `tools/research/agents/prompts.py` -> `build_prompt()`

```python
def build_prompt(role: str, manifest: dict, output_name: str, root=None) -> str:
    if role not in ROLE_GUIDANCE:
        raise ValueError("undefined role")
    if not manifest.get("task_id") or not manifest.get("inputs_sha256"):
        raise ValueError("unbound task/inputs")
    if role in ("GEOMETRY", "REVIEW") and not manifest.get("behavior_evidence_sha256"):
        raise ValueError("missing behavior evidence")
    if role == "REVIEW" and not (manifest.get("candidate_id")
                                  and manifest.get("candidate_sha256")):
        raise ValueError("candidate identity missing")
    # Name the ABSOLUTE output path. A relative path is resolved against the git
    # root, which is the repository, so the Agent would write its answer outside
    # the workspace it is allowed to write in.
    task_id = manifest["task_id"]
    if root is not None:
        destination = str(Path(root) / "agents" / task_id / output_name)
        instruction = ("Write the answer to " + destination + " (this absolute path). "
                       "Write nowhere else.")
    else:
        destination = output_name
        instruction = ("Write the answer to " + destination + " in your working directory.")
    guidance = ROLE_GUIDANCE[role]
    if role in ("MATH_VERIFY", "GRADIENT_VERIFY", "RUNTIME_VERIFY"):
        guidance = guidance + "\n" + UNITTEST_CONTRACT
    if role in ("GEOMETRY", "REVIEW"):
        # The numbers come from the contract compiler, so a prompt can never
        # disagree with the gate that enforces them.
        from ..science import contract as _contract
        # The admission rule changed (issue #120: technical validity only), so the
        # historical quality thresholds were removed from THRESHOLDS and the two
        # Resample numbers survive as REFERENCE_VALUES. The prompt must never
        # invent a gate the contract no longer enforces, and never KeyError when
        # the dict is empty.
        thresholds = getattr(_contract, "THRESHOLDS", {}) or {}
        if ("valid_l1_hard_er_strict_gt" in thresholds
                and "level2_prefixes_strict_gt" in thresholds):
            guidance = guidance + (
                "\nFROZEN_THRESHOLDS: valid_l1_hard_er strictly above "
                f"{thresholds['valid_l1_hard_er_strict_gt']}; "
                f"level2_distinct_prefixes strictly above "
                f"{thresholds['level2_prefixes_strict_gt']}; "
                f"schema_version {_contract.SCHEMA_VERSION}.")
        else:
            reference = getattr(_contract, "REFERENCE_VALUES", {}) or {}
            guidance = guidance + (
                "\nFROZEN_REFERENCE_VALUES (historical reference, NOT admission "
                "thresholds: Stage2 admission is technical validity only, issue #120): "
                "valid_l1_hard_er reference "
                f"{reference.get('valid_l1_hard_er')}; level2_distinct_prefixes reference "
                f"{reference.get('level2_distinct_prefixes')}; "
                f"schema_version {_contract.SCHEMA_VERSION}.")
    return ("ROLE: " + role + "\n" + guidance
            + "\nFROZEN CURRENT-ONLY MANIFEST:\n"
            + json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2)
            + "\n" + instruction
            + " Do not use prior-candidate prompts or any unseen future/test interactions.")
```
