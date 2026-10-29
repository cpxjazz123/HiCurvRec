"""A metric round: propose hyperbolic SID quality metrics, measure their correlation.

This is a different job from a mechanism round, and it reuses the same agent
machinery on purpose -- the real OpenCode subprocess, the isolation, the receipts,
the concurrent phase runner -- while changing the OBJECTIVE and removing every
training phase.

The task: given the 97 confirmed SID+recall pairs (one SID table and the
test_recall@10 of the Stage 3 run that read it), propose a metric computable from
the SID table alone -- plus the frozen Issue #126 behaviour reference and the
TRAINING split, never validation or test interactions -- that is hyperbolic in its
construction and whose Spearman correlation with recall is high.

There is no Stage 2 and no Stage 3 in this round. A metric is judged by one thing:
how well it tracks the recall of tables that already exist.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import agents as agent_contracts
from . import plaintext
from . import pipeline as pipeline_module
from .worker import _run_agent, _run_agents_concurrently

#: What the agents are asked to invent.
METRIC_OBJECTIVE = (
    "Propose a HYPERBOLIC quality metric for an exported Stage-2 Semantic-ID table "
    "that correlates with the Stage-3 test Recall@10 of the run that read it. The "
    "metric is a NUMBER computed from the SID table; nothing is trained, no Stage 2 "
    "and no Stage 3 run is made, and the metric must be computable in under a minute "
    "on 24587 items. It must be judged on 97 SID+recall pairs that already exist."
)

#: What a metric may and may not read. Leakage is the failure that makes a metric
#: look wonderful and mean nothing, so it is stated as a hard boundary.
METRIC_CONTRACT = (
    "A METRIC is IN SCOPE only when all of this holds:\n"
    "  1. HYPERBOLIC CONSTRUCTION: the number uses an explicit curved operation -- "
    "a Poincare or Lorentz distance, exp/log maps, a geodesic or Frechet barycentre "
    "in a ball of fixed negative curvature, a tangent-space comparison that is only "
    "meaningful because of the map. A Euclidean distance in disguise does not "
    "qualify, and the proposal must say where the curvature enters.\n"
    "  2. COMPUTABLE FROM THE TABLE: the inputs are the exported SID table (the "
    "integer code per item and level), optionally the frozen Issue #126 behaviour "
    "reference (source/target representations per item, learned on training pairs "
    "only) and optionally the TRAINING split. Nothing else.\n"
    "  3. NO LEAKAGE: no validation interaction and no test interaction may enter "
    "the number, directly or through a fitted quantity. A metric that reads the "
    "recall, the Stage 3 run, or a validation/test pair is out of scope.\n"
    "  4. FIXED FORM: no free parameter tuned against the 97 recalls. Any constant "
    "must be derived (from the codebook size, the item count, the reference's own "
    "temperature) and stated.\n"
    "  5. ONE NUMBER: the metric returns one scalar per SID table, higher meaning "
    "better. Diagnostics that need a second number to interpret are out of scope.\n"
    "A metric that simply re-expresses the prefix count, the collision rate or the "
    "same-bucket ratio is out of scope: those are already measured and the bucket "
    "count does not track recall (Spearman -0.065 on the 97 pairs)."
)

#: How the answer must be shaped, so the operator can implement it without guessing.
METRIC_PROMPT = (
    "Propose exactly one hyperbolic quality METRIC for exported Semantic-ID tables. "
    "You are one of {agents} independent agents: aim at a different part of the "
    "design space than the others, and do not copy an existing metric.\n"
    "THE OBJECTIVE:\n{objective}\n"
    "THE CONTRACT (a proposal that misses it is not evaluated):\n{contract}\n"
    "WHAT IS ALREADY KNOWN, measured on the 97 pairs (do not repeat these):\n"
    "  - Q_H (the Issue #126 prefix ranking score, level 1) reaches Spearman +0.634;\n"
    "  - bucket count reaches -0.065, same-bucket ratio is weak;\n"
    "  - Q_H at level 2 drops to +0.287 and at level 3 to +0.393, because from level "
    "2 the prefix is nearly an item identifier.\n"
    "You have no write permission: print your answer in the response itself.\n"
    "Write PLAIN TEXT, one block, in exactly this shape:\n"
    "METRIC: <short safe identifier>\n"
    "TITLE: <one line>\n"
    "FORMULA: <the mathematics, with every symbol defined; the evaluator implements "
    "exactly this>\n"
    "INPUTS: <the files or arrays it reads, and nothing more>\n"
    "GEOMETRY: <the curved operation, its curvature, and where it enters>\n"
    "WHY IT SHOULD TRACK RECALL: <the argument, and what it adds over Q_H>\n"
    "COMPUTE: <cost on 24587 items and 256 codes>\n"
    "RISKS: <what could make it fail or be degenerate>\n"
    "FAILURE MODE: <the SID table on which it must be worst, and why>\n"
    "END METRIC"
)


def run_metric_ideation(root: Path, *, agents: int = None) -> dict:
    """Five (or ``agents``) independent agents, one metric each, persisted as they land."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    count = int(agents or pipeline_module.IDEA_AGENTS)
    prompt_base = METRIC_PROMPT.format(agents=count, objective=METRIC_OBJECTIVE,
                                       contract=METRIC_CONTRACT)

    def propose(index):
        task_id = "metric_ideation_" + str(index)
        prompt = (prompt_base + "\n\nYour answer must be a different metric from the "
                  "other " + str(count - 1) + " agents, which you cannot see.")
        return (task_id, _run_agent(root, task_id, prompt, "metric.txt",
                                    plaintext.parse_metric_ideas,
                                    role="ideate", read_paths=[],
                                    allowed_roots=[root]))

    results = _run_agents_concurrently([(index,) for index in range(1, count + 1)],
                                       propose)
    metrics = []
    for task_id, answer in results:
        for metric in (answer or {}).get("metrics", []):
            metric = dict(metric)
            metric["proposed_by"] = task_id
            metrics.append(metric)
    record = {"objective": METRIC_OBJECTIVE, "contract": METRIC_CONTRACT,
              "agents": count, "at": time.time(), "metrics": metrics}
    (root / "metric_ideas.json").write_text(json.dumps(record, indent=2),
                                            encoding="utf-8")
    return record
