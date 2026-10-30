# The round is blocked on the model provider's quota

Started: `research_round_start` with issue 121 and the full-chain objective.
Round id: `round_20261031_015906_400`. Worker pid 105099.

## What happened

The round worker launched the five ideation agents and **all five failed in about
eight seconds** with `failure_class: quota` and `exit_code: 1`, and the state
records `round_failed` with `the phase is incomplete: 1 (unknown); 2 (unknown);
3 (unknown); 4 (unknown); 5 (unknown)`.

Evidence, from `agents/ideation_1/execution.json`:

* `failure_class`: `quota`
* `exit_code`: `1`
* `elapsed_seconds`: `8.113`
* `timed_out`: `False`, `cancelled`: `False`, `deadline_passed`: `False`
* `sandbox`: bwrap mount namespace, `network: shared (model provider)`

A direct probe confirms it:

```
$ opencode run --standalone --model opencode-go/deepseek-v4.1-flash "say ok"
Error: Go usage limit exceeded
```

## Why this blocks the whole chain

`tools/research/settings.py` sets `AGENT_MODEL = "opencode-go/deepseek-v4.1-flash"`
and that provider reports `Go usage limit exceeded`. Every agent phase of the
framework runs through it: ideation, the pre-score probe, scoring, and the five
verifiers. The repo rules also require four independently authored test suites
before any formal Stage2 run (`verification_check`, §26 to §31), and those
authors are agents too. So the block is not just the ideation phase: **no Stage2
or Stage3 can be admitted while the provider is out of quota**.

Per §19c and §70 a quota failure pauses the service and is NOT retried in a
loop, so this round stays where it is until the quota is restored or the model
is changed.

## What is needed to continue

1. restore the quota for `opencode-go/deepseek-v4.1-flash`, or
2. point `AGENT_MODEL` (a source constant, no CLI flag) at a provider with
   quota, and re-run `research_round_start` with the same objective.

Nothing else in the chain is blocked: the baseline tree
`stage2_RQ-VAE/TIGER_RQ-VAE` is intact, the four-GPU recipe is frozen, and the
standing GPU authorisation (§59) is in place. The work that remains after the
quota returns is the same round: ideation, the evidence probe, scoring, one
selected mechanism, the implementation by the main agent, five verifiers, then
Stage2 (port 50202) and Stage3 (port 50201).
