"""The end-to-end dry run of the hyperbolic-only campaign (issue #110, WP5).

The dry run walks the whole flow on a SYNTHETIC candidate: the literature search, the
prior-art check, the mathematics review, the implementation, the CGV, one probe, the cold
Stage-2, the frozen evaluation, the Stage-3, the analysis and the resume.

Everything that would spend a GPU is MOCKED, and the mocks are marked as such in the
artefact the run writes. A dry run is not a scientific success and is never recorded as
one: the success predicate is asked, it refuses, and the refusal is in the output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where the dry run writes its artefact.
DRY_RUN_DIR = REPO_ROOT / "results" / "hyperbolic_research_runs" / "dry_run"

#: The stages the dry run walks, in the order the issue lists them.
STAGES = (
    "search",
    "prior_art",
    "math_review",
    "implementation",
    "cgv",
    "probe",
    "stage2",
    "frozen_evaluation",
    "stage3",
    "analysis",
    "resume",
)

#: The stages that are mocked, because they would otherwise spend the machine.
MOCKED_STAGES = ("probe", "stage2", "frozen_evaluation", "stage3")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def synthetic_proposal() -> Dict[str, Any]:
    """A synthetic hyperbolic proposal, with the seven-group contract and the geometry."""
    return {
        "mechanism_id": "dry_run_synthetic",
        "family": "hyperbolic_history",
        "equation": "L = sum_i m_i * busemann_xi(source_i, codebook_xi) + 0.05 * ||codebook||^2",
        "symbols": {
            "m_i": "the mask of the pair",
            "busemann_xi": "the Busemann function of the base point xi",
        },
        "parameters": {"c": 1.0, "weight": 0.05},
        "gradient_targets": {"source": "zero", "target": "zero", "codebook": "nonzero"},
        "target_paths": ["codebook"],
        "behaviour_signal": "the transition pair",
        "codebook_interaction": "the codebook moves along the horosphere of the base point",
        "encoder_interaction": "none",
        "risk": "the horosphere may not order the items the way the behaviour does",
        "falsifier": "the exported level-0 usage does not change",
        "why_not_the_measured_failures": "it is a geodesic term, not a pairwise pull",
        "hyperbolic_model": "poincare_ball",
        "hyperbolic_operator": "busemann",
        "curvature": -1.0,
        "basepoints": "the origin",
        "literature_ids": [],
        "cgv_contract": {
            "mechanism": {
                "id": "dry_run_synthetic",
                "type": "auxiliary_term",
                "entrypoints": ["mechanism_loss"],
            },
            "model": {"adapter": "rqvae", "parameter_groups": {"codebook": "codebook"}},
            "forward_contract": {"observables": ["loss"], "invariants": []},
            "gradient_contract": {
                "mode": "invariants",
                "target": "codebook",
                "invariants": [],
            },
            "update_contract": {"optimizer": "adam", "lr": 1e-3, "clip_norm": 5.0},
            "distributed_contract": {"world_size": 4, "aggregation": "mean"},
            "runtime_contract": {"entrypoint": "train_rqvae.py", "observe": ["loss"]},
        },
    }


def synthetic_source() -> str:
    """The source of the synthetic mechanism, with a genuinely hyperbolic operator in it."""
    return '''
import torch


def _busemann(x, xi, c):
    """The Busemann function of the base point xi, in the Lorentz model."""
    dxy = ((x - xi) ** 2).sum(-1)
    inner = 1.0 - 1.0 / c
    numerator = dxy - inner * (1.0 - (x ** 2).sum(-1))
    denominator = dxy - inner * (1.0 - (xi ** 2).sum(-1))
    return torch.log((numerator / denominator).clamp(min=1e-15)) / c ** 0.5


def mechanism_loss(source, target, codebook, mask, **parameters):
    c = float(parameters.get("c", 1.0))
    weight = float(parameters.get("weight", 0.0))
    base = torch.zeros_like(codebook[0])
    distance = _busemann(source.unsqueeze(1), base.unsqueeze(0), c)
    picked = distance.argmin(dim=-1)
    gathered = distance.gather(1, picked.unsqueeze(1)).squeeze(1)
    return (gathered * mask).mean() + weight * (codebook ** 2).sum()
'''


class _FakeProvider:
    """A provider that answers, so the dry run never touches the network."""

    name = "dry_run_fake"
    version = "v1"

    def __init__(self) -> None:
        self.asked: List[str] = []

    def search(self, query: str, *, limit: int = 3):
        from tools.genrec_agents.literature import PaperRecord, SearchResponse, now_utc

        self.asked.append(query)
        response = SearchResponse(
            provider=self.name,
            version=self.version,
            query=query,
            requested_utc=now_utc(),
        )
        response.records = [
            PaperRecord(
                identifier="10.1000/dry-run",
                title="A synthetic reference for the dry run",
                authors=("Dry Run",),
                year=2026,
                url="https://doi.org/10.1000/dry-run",
                source=self.name,
                access_level="verified_abstract_only",
                retrieved_utc=response.requested_utc,
            )
        ]
        return response


def run_dry_run(out_dir: Path = DRY_RUN_DIR) -> Dict[str, Any]:
    """Walk the whole flow on a synthetic candidate and write what happened."""
    from tools.genrec_agents import implementation, literature, prior_art, success
    from tools.genrec_agents.hyperbolic_campaign import Campaign, Candidate
    from tools.hyperbolic_verifier import gates

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal = synthetic_proposal()
    source = synthetic_source()
    proposal_sha = implementation.sha256_bytes(
        json.dumps(proposal, sort_keys=True).encode("utf-8")
    )
    implementation_sha = implementation.sha256_bytes(source.encode("utf-8"))
    protocol_sha = implementation.sha256_bytes(b"dry-run-protocol-v1")
    stages: Dict[str, Any] = {}

    # 1. the literature search
    provider = _FakeProvider()
    agent = literature.LiteratureAgent(out_dir=out_dir / "literature", providers=[provider])
    review = agent.search("hyperbolic semantic ids recommendation", limit=3)
    stages["search"] = {
        "state": review["state"],
        "papers": review["papers"],
        "provider": provider.name,
        "mocked": False,
    }

    # 2. the prior-art check, against the real registry
    registry = prior_art.load_registry()
    verdict = prior_art.check_novelty(
        proposal, registry, implementation_source=source
    )
    stages["prior_art"] = {
        "allowed": verdict.allowed,
        "reason": verdict.reason,
        "registry_entries": len(registry["entries"]),
        "mocked": False,
    }

    # 3. the mathematics review: the H gates on the synthetic source
    certificate = gates.run_gates(
        proposal,
        source,
        proposal_sha256=proposal_sha,
        code_tree_sha256=implementation_sha,
        gradient_evidence={
            "declared_targets": ["codebook"],
            "reached_targets": ["codebook"],
            "finite": True,
        },
        activation_evidence={
            "operator_enabled": True,
            "changed_fraction": 0.42,
            "distinct_level_codes": 240,
        },
    )
    stages["math_review"] = {
        "gates": {name: result.state for name, result in certificate.gates.items()},
        "passed": certificate.passed,
        "mocked": False,
    }

    # 4. the implementation, against the frozen proposal
    tree = out_dir / "tree"
    tree.mkdir(parents=True, exist_ok=True)
    module = tree / "mechanism_dry_run_synthetic.py"
    module.write_text(source, encoding="utf-8")
    request = implementation.ImplementationRequest(
        mechanism_id=proposal["mechanism_id"],
        proposal=proposal,
        proposal_sha256=proposal_sha,
        tree=tree,
        frozen_protocol_hash=protocol_sha,
    )
    implemented = implementation.implement(
        request,
        proposal,
        written=[module],
    )
    stages["implementation"] = {
        "status": implemented.status,
        "implementation_sha256": implemented.implementation_sha256,
        "mocked": False,
    }

    # 5. the CGV: the dry run records the six gate names it must produce. The CGV itself
    #    needs the real trainer, so the dry run states that the gates are MOCKED here rather
    #    than claiming they passed.
    stages["cgv"] = {
        "gates": list(success.CGV_GATES),
        "state": "mocked",
        "note": "the real CGV runs the trainer; the dry run only proves the path is wired",
        "mocked": True,
    }

    # 6. one probe
    stages["probe"] = {
        "state": "mocked",
        "g_h1": 0.1234,
        "note": "a real probe costs a GPU run; the dry run records the call path",
        "mocked": True,
    }

    # 7. the cold Stage-2
    stages["stage2"] = {
        "state": "mocked",
        "epochs": 3000,
        "seed": 42,
        "note": "no cold run is performed by a dry run",
        "mocked": True,
    }

    # 8. the frozen evaluation
    stages["frozen_evaluation"] = {
        "state": "mocked",
        "er": 16.42,
        "pos_share": 0.079,
        "neg_share": 0.0048,
        "d2": 0.04,
        "note": "the frozen pairs are not read again for a dry run",
        "mocked": True,
    }

    # 9. the Stage-3
    stages["stage3"] = {
        "state": "mocked",
        "recall": 0.0610,
        "ndcg": 0.0330,
        "note": "no Stage-3 is performed by a dry run",
        "mocked": True,
    }

    # 10. the analysis, and the success predicate, which must refuse
    from tools.genrec_agents import scientific_analysis

    analysis = scientific_analysis.failure_analysis(
        [
            {
                "mechanism_id": proposal["mechanism_id"],
                "kind": "stage2",
                "metrics": {
                    "frozen_evaluation": {
                        "hard_l1_er": 16.42,
                        "pos_share": 0.079,
                        "neg_share": 0.0048,
                        "d2": 0.04,
                        "d2_block": {"denominator_positive": 500, "flags": ["low_support"]},
                    },
                    "stage3": {"test_recall@10": 0.0610, "test_ndcg@10": 0.0330},
                    "gh1_probe": {"g_h1": 0.1234},
                },
            }
        ],
        {},
    )
    stages["analysis"] = {
        "failure_kinds": analysis["failure_kinds"],
        "questions": len(scientific_analysis.research_questions(analysis)),
        "mocked": False,
    }

    # the success predicate, asked and refused
    win = success.is_verified_triple_win(
        {
            "proposal_sha256": proposal_sha,
            "implementation_sha256": implementation_sha,
            "frozen_protocol_hash": protocol_sha,
            "seed": 42,
            "hyperbolic_certificate": certificate.as_dict(),
            "cgv_certificate": {gate: {"state": "pass"} for gate in success.CGV_GATES},
            "frozen_evaluation": {"hard_l1_er": 16.42, "pos_share": 0.079, "neg_share": 0.0048},
            "stage3": {
                "test_recall@10": 0.0610,
                "test_ndcg@10": 0.0330,
                "complete": True,
            },
        }
    )
    stages["success_predicate"] = {
        "won": win.won,
        "reason": win.reason,
        "mocked": False,
    }

    # 11. the state machine moves the candidate through the flow, and resumes
    campaign = Campaign(namespace=out_dir / "campaign")
    candidate = Candidate(
        mechanism_id=proposal["mechanism_id"],
        proposal_sha256=proposal_sha,
        family=proposal["family"],
        implementation_sha256=implementation_sha,
        frozen_protocol_hash=protocol_sha,
    )
    campaign.add(candidate)
    for to_state in ("reviewed", "implemented", "screened", "sealed"):
        campaign.transition(candidate.mechanism_id, to_state)
    identity = campaign.reserve_experiment(candidate.mechanism_id, "stage2")
    campaign.transition(candidate.mechanism_id, "stage2_reserved")
    campaign.transition(candidate.mechanism_id, "stage2_running")
    campaign.transition(candidate.mechanism_id, "stage2_complete")
    campaign.transition(candidate.mechanism_id, "evaluated")
    summary = campaign.summary()
    stages["resume"] = {
        "states": summary["states"],
        "reserved_experiments": summary["official_experiments"],
        "identity": identity,
        "mocked": False,
    }

    report = {
        "generated_utc": now_utc(),
        "dry_run": True,
        "is_scientific_success": False,
        "warning": (
            "this is a dry run on a synthetic candidate: the GPU stages are mocked, and it is "
            "not a scientific success and must never be recorded as one"
        ),
        "proposal_sha256": proposal_sha,
        "implementation_sha256": implementation_sha,
        "frozen_protocol_hash": protocol_sha,
        "stages": stages,
        "stage_order": list(STAGES),
        "mocked_stages": list(MOCKED_STAGES),
    }
    (out_dir / "dry_run.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["DRY_RUN_DIR", "MOCKED_STAGES", "STAGES", "run_dry_run", "synthetic_proposal"]
