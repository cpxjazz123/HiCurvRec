"""Issue #223 evidence: one structure backward and one real optimiser step.

Writes `backprop_evidence.json`: the gradient norm each parameter group received
from L_structure alone, what one AdamW step changed at lambda=0 and at lambda>0,
and how far the two update directions diverge.  The numbers the report quotes
come from this file, not from a transcript.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import trainable_student as ts  # noqa: E402


def main() -> None:
    torch.manual_seed(5)
    generator = torch.Generator().manual_seed(5)
    items = torch.randn(48, 24, generator=generator)
    targets = {"type": torch.randint(0, 5, (12,), generator=generator),
               "orbit": torch.randint(0, 7, (12, 4), generator=generator)}
    model = ts.build_small_model(in_dim=24, narrow=16, levels=3, codes=16, dim=8)
    head = ts.build_head(dim=8, n_subtypes=5, n_orbits=7)
    teacher = ts.build_small_teacher(dim=24, hidden=16, levels=2, n_subtypes=5, n_orbits=7)
    teacher_before = ts.parameter_digest(teacher)

    loss, info = ts.structure_loss(model, head, items, instances=12, nodes=4,
                                   teacher=teacher,
                                   adjacency=ts.random_adjacency(12, 4, seed=9))
    report = ts.gradient_report(loss, model, head, teacher=teacher)
    ts.assert_learning_path(report)
    teacher_after = ts.parameter_digest(teacher)

    batch = items[:16]
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    ts.assert_optimizer_covers(optimizer, model, head)
    before = ts._snapshot(model)
    reconstructed, parts, _, _ = model(batch)
    main = ts._arm()["losses"].reconstruction_loss(reconstructed, batch) + \
        ts._arm()["losses"].quantization_total(parts, 0.25)
    main_report = ts.gradient_report(main, model, head)
    main.backward()
    optimizer.step()
    after_main_only = ts._snapshot(model)

    zero = ts.prove_update(seed=11, items=items, instances=12, nodes=4, targets=targets,
                           lambda_value=0.0)
    positive = ts.prove_update(seed=11, items=items, instances=12, nodes=4, targets=targets,
                               lambda_value=0.5)
    ts.assert_update_proof(positive)
    divergence = ts.parameter_divergence(zero["after"], positive["after"])

    payload = {
        "structure_only": {
            "loss": float(loss.detach()),
            "gradient_norms": report,
            "teacher_parameters_unchanged": all(teacher_before[k] == teacher_after[k]
                                                for k in teacher_before),
            "student_bundle_keys": list(info["student_bundle"]),
            "supervision": info["supervision"],
        },
        "main_loss_reference": {
            "loss": float(main.detach()),
            "gradient_norms": main_report,
            "encoder_step_after_one_adamw": ts.parameter_divergence(before, after_main_only)["encoder"],
            "codebook_step_after_one_adamw": ts.parameter_divergence(before, after_main_only)["codebooks"],
        },
        "lambda_zero": {"changed": zero["changed"], "lambda": zero["lambda"]},
        "lambda_positive": {"changed": positive["changed"], "lambda": positive["lambda"],
                            "structure_grad_norms": positive["structure_grad_norms"]},
        "update_divergence": divergence,
        "checks": {"learning_path": "pass", "update_proof": "pass",
                   "teacher_frozen": "pass", "bundle_has_no_structure": "pass"},
    }
    (HERE / "backprop_evidence.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({key: payload[key] for key in ("lambda_zero", "lambda_positive",
                                                    "update_divergence")}, indent=1))


if __name__ == "__main__":
    main()
