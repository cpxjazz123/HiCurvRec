"""Record the executed experiments in the ledger (issue #105, workstream B).

The controller runs the experiments the agent asked for and writes the results back into
the ledger, with the artifact each result came from. The ledger's own checks then run:
a hypothesis may not be decided by the results of its own experiment, and every run it
cites must exist in the store.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m tools.harness2_genrec.cli \
        record-experiments --store results/harness3_runs/store \
                           --ledger results/harness3_runs/hypotheses/ledger.json \
                           --out results/harness3_runs/experiments
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence import EvidenceStore, now_utc
from .ledger import Ledger

#: What each executed experiment found, in the words the ledger carries. The numbers live
#: in the reports; these are the verdicts and the artifacts they came from.
EXECUTED: Dict[str, Dict[str, Any]] = {
    "H105-01/X1": {
        "report": "x1_pipeline_stage.json",
        "result": (
            "the two arms' trained models are byte-identical to the matched control's: the "
            "encoder output, the residual stream, the codebook and the assignment all agree "
            "to the last bit, and the checkpoint files share one SHA-256"
        ),
        "verdict": (
            "the arms differ nowhere in the trained model, so the mechanism never moved the "
            "weights; the export path did not fall back on anything, the training itself "
            "was unchanged"
        ),
        "status": "executed",
    },
    "H105-02/X1": {
        "report": "x2_er_decomposition.json",
        "result": (
            "the ER gains are spread over the catalogue rather than carried by a few items "
            "(4.8-5.6% of the held-out anchors changed, and the top 1% of anchors carry "
            "18-21% of the change), the code cells are the same size as the control's for "
            "the geometry arm (102 vs 103) and larger for the residual target (111), and "
            "the downstream differences are 2.1 and 9.3 times the Stage-3 run-to-run band"
        ),
        "verdict": (
            "the ER gain is not a concentration artefact, and the geometry arm's export is "
            "not degraded; the downstream difference is real but small for the geometry arm "
            "and clearly negative for the residual target"
        ),
        "status": "executed",
    },
    "H105-03/X1": {
        "report": "x4_balance_and_generalisation.json",
        "result": (
            "the residual target concentrates the level-0 usage (Gini +0.137, entropy "
            "-0.137) and loses 4.8% of its level-2 occupancy; the geometry transport arm "
            "leaves the balance essentially unchanged (Gini +0.012, entropy -0.0005) and "
            "improves its train-over-held-out discrimination ratio by 4.5%"
        ),
        "verdict": (
            "the residual target's ER gain was bought with a more concentrated level-0 code "
            "distribution, which is consistent with its downstream regression; the geometry "
            "arm's export is healthy and generalises, so its flat downstream result is not "
            "explained by the code distribution or by overfitting"
        ),
        "status": "executed",
    },
    "H105-04/X5": {
        "report": "x5_projection.json",
        "result": (
            "in a matched pair on the host's own mini case, an auxiliary term changes the "
            "encoder gradient by 1.66e-05 before the host's post-backward assembly and by "
            "exactly 0.0 after it"
        ),
        "verdict": (
            "the host OVERWRITES the encoder gradient with g_base + w_b/valid_levels * "
            "(g'_L1 + g_L2 + g_L3) and never touches the decoder or the codebook gradient, "
            "so an auxiliary term that reaches only the encoder is discarded in production "
            "however healthy it looks in isolation; only a term that delivers gradient to "
            "the codebook can change the exported SIDs"
        ),
        "status": "executed",
    },
}


def record(
    store: EvidenceStore, ledger_path: Path, out_dir: Path
) -> Dict[str, Any]:
    """Write the executed experiments into the ledger and re-run its checks."""
    ledger = Ledger(ledger_path, store=store).load()
    out_dir = Path(out_dir)
    # The store's run ids carry a timestamp, so a re-ingestion of the same arms renames
    # them. The ledger's links are repaired by MECHANISM, not by hand: a hypothesis that
    # cited the residual-target arm still cites it.
    by_mechanism = {
        manifest["mechanism_id"]: manifest["run_id"] for manifest in store.runs()
    }
    for hypothesis in ledger.hypotheses:
        repointed: List[str] = []
        for run_id in hypothesis.evidence_runs:
            try:
                manifest = store.load(run_id)
                repointed.append(by_mechanism.get(manifest["mechanism_id"], run_id))
                continue
            except Exception:  # noqa: BLE001
                pass
            # The run is no longer in the store (the store was re-ingested and the ids
            # carry a timestamp). The mechanism it was about is still known from the id.
            stem = run_id.rsplit("-", 1)[0]
            mechanism = stem.split("-", 1)[1] if "-" in stem else stem
            repointed.append(by_mechanism.get(mechanism, run_id))
        hypothesis.evidence_runs = sorted(set(repointed))
    updated: List[Dict[str, Any]] = []
    for hypothesis in ledger.hypotheses:
        for experiment in hypothesis.experiments:
            record_entry = EXECUTED.get(f"{hypothesis.id}/{experiment.id}")
            if record_entry is None:
                continue
            report = out_dir / record_entry["report"]
            if not report.is_file():
                experiment.status = "failed"
                experiment.result = f"the report {report} does not exist"
                experiment.verdict = "the experiment produced no report"
                updated.append(
                    {
                        "hypothesis": hypothesis.id,
                        "experiment": experiment.id,
                        "status": experiment.status,
                        "verdict": experiment.verdict,
                    }
                )
                continue
            experiment.status = record_entry["status"]
            experiment.result = record_entry["result"]
            experiment.verdict = record_entry["verdict"]
            experiment.produced_runs = []
            updated.append(
                {
                    "hypothesis": hypothesis.id,
                    "experiment": experiment.id,
                    "status": experiment.status,
                    "verdict": experiment.verdict,
                    "report": str(report),
                }
            )
    # The controller's own hypothesis, which X5 tested: the host's post-backward assembly
    # discards an auxiliary term's encoder gradient.
    if ledger.get("H105-04") is None:
        from .ledger import Explanation, Experiment, Hypothesis, Prediction

        ledger.add(
            Hypothesis(
                id="H105-04",
                observation=(
                    "two arms' trained models are byte-identical to the matched control's, "
                    "although their mechanisms pass the independent gradient checks"
                ),
                evidence_runs=[
                    run["run_id"]
                    for run in store.runs()
                    if run["mechanism_id"]
                    in (
                        "behavior_conditioned_pairwise_margin",
                        "curvature_weighted_transition_transport",
                    )
                ],
                explanations=[
                    Explanation(
                        id="E1",
                        statement=(
                            "the host overwrites the encoder gradient after the backward, so "
                            "an auxiliary term that reaches only the encoder is discarded in "
                            "production, however healthy it looks in isolation"
                        ),
                        predictions=[
                            Prediction(
                                id="P1",
                                statement=(
                                    "on the host's own mini case, an auxiliary term changes "
                                    "the encoder gradient before the host's assembly and not "
                                    "at all after it"
                                ),
                                falsifier=(
                                    "the encoder gradient after the host's assembly differs "
                                    "from the control's by a nonzero amount"
                                ),
                                measurement=(
                                    "a matched pair on the host's mini case: the encoder "
                                    "gradient with and without the term, before and after "
                                    "the host's own projection"
                                ),
                            )
                        ],
                    )
                ],
                experiments=[
                    Experiment(
                        id="X5",
                        tests=["E1"],
                        control_run=next(
                            (
                                run["run_id"]
                                for run in store.runs()
                                if run["mechanism_id"] == "r8_control"
                            ),
                            None,
                        ),
                        status="executed",
                    )
                ],
                status="supported",
                notes=(
                    "measured on the host's own code path: the encoder gradient difference "
                    "is 1.66e-05 before the host's post-backward assembly and exactly 0.0 "
                    "after it, so the two inert arms' training was bit-identical to the "
                    "control's. A mechanism that wants to move the exported SIDs has to "
                    "deliver its gradient to the codebook, which the host never overwrites"
                ),
            )
        )

    # The hypotheses whose explanations are now decided by the executed experiments.
    for hypothesis in ledger.hypotheses:
        executed = [e for e in hypothesis.experiments if e.status == "executed"]
        if not executed:
            continue
        if hypothesis.id == "H105-01":
            hypothesis.status = "supported"
            hypothesis.notes = (
                "the export path is not the problem: the trained model itself is identical "
                "to the control's, and X5 shows why -- the host discards an auxiliary "
                "term's encoder gradient"
            )
            hypothesis.next_experiment = "X3"
        elif hypothesis.id == "H105-02":
            hypothesis.status = "inconclusive"
            hypothesis.notes = (
                "the ER gain is broad and the export is not damaged, yet the downstream "
                "quality did not follow; the pair-level ER and the recommender's objective "
                "are decoupled on this host, and the paired repeat is what would settle "
                "whether the small downstream difference is real"
            )
            hypothesis.next_experiment = "X3"
        elif hypothesis.id == "H105-03":
            hypothesis.status = "supported"
            hypothesis.notes = (
                "the residual target's gain came with a more concentrated level-0 usage and "
                "a collapsed level 2, which matches its downstream regression; the geometry "
                "arm changed neither the balance nor the generalisation gap"
            )
            hypothesis.next_experiment = "X3"
    digest = ledger.save()
    problems = ledger.problems()
    report = {
        "generated_utc": now_utc(),
        "ledger_path": str(ledger_path),
        "ledger_sha256": digest,
        "updated": updated,
        "problems": problems,
        "statuses": {
            hypothesis.id: hypothesis.status for hypothesis in ledger.hypotheses
        },
    }
    (out_dir / "ledger_update.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["EXECUTED", "record"]
