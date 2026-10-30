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

## Provider survey (2026-10-31, after the failure)

Four authenticated providers were probed directly. All four are unusable, so no
agent phase of the framework can run and, because the four independently
authored test suites that gate Stage2 are written by agents, no formal training
can be admitted either.

| provider | model probed | result |
| --- | --- | --- |
| OpenCode Go | `opencode-go/deepseek-v4.1-flash` (the current `AGENT_MODEL`) | `Error: Go usage limit exceeded` |
| Hugging Face (`HF_TOKEN` in the environment) | `huggingface/deepseek-ai/DeepSeek-V4.1-Flash` | `Provider request failed with HTTP 402` (payment required) |
| MiniMax (`MINIMAX_API_KEY` in the environment) | `minimax-cn/MiniMax-M3` | `Error: invalid api key` |
| OpenAI (OAuth credential stored) | `openai/gpt-4o` | `Error: Model unavailable`; the model list contains no `openai/*` entry at all |

`opencode models` lists 149 models across the providers, but a listing is not a
credential.

## The single change that unblocks it

`tools/research/settings.py` line 34 sets the model every agent phase uses:

```python
AGENT_MODEL = "opencode-go/deepseek-v4.1-flash"
```

Point it at a model whose provider has working credentials, then reload the MCP
server (a change under `tools/` is not picked up by the running server, §52 to
§57):

```bash
OLD=$(pgrep -f 'genrec-mcp/server.py' | head -1); kill -TERM "$OLD"
for i in $(seq 10); do sleep 3; NEW=$(pgrep -f 'genrec-mcp/server.py' | head -1); \
  [ -n "$NEW" ] && [ "$NEW" != "$OLD" ] && break; done
```

and start a fresh round with the same objective. Everything else is in place:
the objective text (the full chain, the measured evidence it must respect, the
0.065 target and the §19g single-run constraints), the intact baseline tree
`stage2_RQ-VAE/TIGER_RQ-VAE`, the frozen four-GPU recipe and the standing GPU
authorisation (§59).

### Two further probes, both on the stored OpenAI OAuth credential

`opencode models` does list `openai/*` entries (gpt-5.5, 5.6, 6 and codex
variants), but the credential behind them is a ChatGPT/Codex account and it
refuses them:

```
$ opencode run --standalone --model openai/gpt-5.5 "reply with exactly: ok"
Error: model_not_found: The 'gpt-5.5' model is not supported when using Codex
with a ChatGPT account.

$ opencode run --standalone --model openai/gpt-5.3-codex-spark "reply with exactly: ok"
Error: model_not_found: The 'gpt-5.3-codex-spark' model is not supported when
using Codex with a ChatGPT account.
```

So the six probes across four providers all fail, and they fail for four
different reasons (usage limit, payment required, invalid key, model not
available to this account). Guessing further model names is not a fix: what is
missing is a working credential.
