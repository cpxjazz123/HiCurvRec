"""Issue #223: the structure loss must move the REAL encoder and the REAL codebooks.

These tests are deliberately written to fail on the pre-#223 state (where the
structure head read a frozen SID): each one either drives a real optimiser step
through the arm's quantiser or injects one of the ways the wiring can silently
be faked -- a detached quantiser, a frozen codebook, an optimiser that does not
hold the codebook parameters -- and requires the detector to reject it.

The model is the arm's own `RQVAE`, the quantiser its own `RQLayer`, the loss
functions its own registry; only the widths are shrunk so a CPU test runs in
seconds.  The smoke run uses the frozen 768 -> 512 -> 256 -> 128 -> 32 widths.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import trainable_student as ts  # noqa: E402


def _small_model(seed: int = 3):
    torch.manual_seed(seed)
    return ts.build_small_model(in_dim=24, narrow=16, levels=3, codes=16, dim=8)


def _batch(seed: int = 5, instances: int = 4, nodes: int = 3):
    generator = torch.Generator().manual_seed(seed)
    items = torch.randn(instances * nodes, 24, generator=generator)
    targets = {
        "type": torch.randint(0, 5, (instances,), generator=generator),
        "orbit": torch.randint(0, 7, (instances, nodes), generator=generator),
    }
    return items, targets, instances, nodes


def _head(model, seed: int = 7):
    torch.manual_seed(seed)
    return ts.build_head(dim=8, n_subtypes=5, n_orbits=7)


def test_structure_gradients_reach_the_encoder_and_all_three_codebooks():
    model = _small_model()
    head = _head(model)
    items, targets, instances, nodes = _batch()
    loss, info = ts.structure_loss(model, head, items, instances=instances, nodes=nodes,
                                   targets=targets)
    assert torch.isfinite(loss)
    report = ts.gradient_report(loss, model, head)
    ts.assert_learning_path(report)
    assert len(report["codebooks"]) == 3, "all three codebooks must be checked"
    for level, norm in enumerate(report["codebooks"]):
        assert norm > 0.0, f"codebook level {level} received no gradient from L_structure"
    assert report["encoder"] > 0.0, "the encoder received no gradient from L_structure"
    assert report["head"] > 0.0


def test_teacher_targets_are_stop_gradient_and_the_teacher_does_not_move():
    model = _small_model()
    head = _head(model)
    teacher = ts.build_small_teacher(dim=24, hidden=16, levels=2, n_subtypes=5, n_orbits=7)
    before = ts.parameter_digest(teacher)
    items, _, instances, nodes = _batch()
    loss, info = ts.structure_loss(model, head, items, instances=instances, nodes=nodes,
                                   teacher=teacher, adjacency=ts.random_adjacency(instances, nodes, seed=9))
    report = ts.gradient_report(loss, model, head, teacher=teacher)
    ts.assert_learning_path(report)
    assert report["teacher"] == 0.0, "the teacher received a gradient"
    after = ts.parameter_digest(teacher)
    assert all(before[name] == after[name] for name in before), "the teacher moved"


def test_a_real_optimiser_step_changes_the_encoder_and_every_codebook():
    items, targets, instances, nodes = _batch()
    lambda_value = 0.5
    positive = ts.prove_update(seed=11, items=items, instances=instances, nodes=nodes,
                               targets=targets, lambda_value=lambda_value)
    ts.assert_update_proof(positive)
    for group in ("encoder", "codebooks"):
        assert positive["changed"][group], f"{group} did not change under lambda={lambda_value}"


def test_lambda_zero_keeps_the_structure_gradient_out_and_still_moves_the_quantiser():
    items, targets, instances, nodes = _batch()
    zero = ts.prove_update(seed=11, items=items, instances=instances, nodes=nodes,
                           targets=targets, lambda_value=0.0)
    assert zero["structure_grad_norms"]["encoder"] == 0.0, \
        "lambda=0 must contribute nothing to the encoder"
    assert zero["changed"]["codebooks"], "the reconstruction/VQ path must still move the codebooks"
    positive = ts.prove_update(seed=11, items=items, instances=instances, nodes=nodes,
                               targets=targets, lambda_value=0.5)
    diverge = ts.parameter_divergence(zero["after"], positive["after"])
    assert diverge["encoder"] > 1e-8, "adding L_structure must change the encoder update direction"


def test_a_detached_quantiser_is_rejected():
    model = _small_model()
    head = _head(model)
    items, targets, instances, nodes = _batch()
    with pytest.raises(ts.LearningPathError):
        loss, _ = ts.structure_loss(model, head, items, instances=instances, nodes=nodes,
                                    targets=targets, detach_quantised=True)
        ts.assert_learning_path(ts.gradient_report(loss, model, head))


def test_a_frozen_codebook_is_rejected():
    model = _small_model()
    head = _head(model)
    items, targets, instances, nodes = _batch()
    loss, _ = ts.structure_loss(model, head, items, instances=instances, nodes=nodes,
                                targets=targets)
    for parameter in ts.codebook_parameters(model).values():
        parameter.requires_grad_(False)
    with pytest.raises(ts.LearningPathError):
        ts.assert_learning_path(ts.gradient_report(loss, model, head))


def test_an_optimiser_without_the_codebooks_is_rejected():
    model = _small_model()
    head = _head(model)
    optimizer = torch.optim.AdamW(list(model.encoder.parameters()) + list(head.parameters()), lr=1e-3)
    with pytest.raises(ts.LearningPathError):
        ts.assert_optimizer_covers(optimizer, model, head)


def test_hard_forward_equals_the_codebook_sum_and_the_tokens_match_get_indices():
    # 32 items > 16 codewords: below that the assignment floor rotates through the
    # least-used code words with a persistent per-layer counter, so two consecutive
    # calls legitimately differ and the export comparison would be meaningless.
    model = _small_model()
    items, _, instances, nodes = _batch(instances=8, nodes=4)
    path = ts.quantised_student_input(model, items, instances=instances, nodes=nodes)
    with torch.no_grad():
        reference = ts.hard_cumulative(path.tokens.reshape(-1, path.tokens.shape[-1]),
                                       [layer.get_code_embs() for layer in model.rq.vq_layers])
        indices = model.get_indices(items)
    assert torch.equal(path.tokens.reshape(-1, path.tokens.shape[-1]), indices), \
        "the structure path quantised different codes than the export path"
    assert torch.allclose(path.q.reshape(-1, path.q.shape[-1]), reference, atol=1e-6), \
        "the hard-forward value is not the sum of the picked code words"


def test_the_student_bundle_carries_no_structure():
    model = _small_model()
    head = _head(model)
    items, targets, instances, nodes = _batch()
    _, info = ts.structure_loss(model, head, items, instances=instances, nodes=nodes,
                                targets=targets)
    ts.assert_no_structure(info["student_bundle"])


def test_the_codebook_parameters_are_the_raw_level_parameters_not_a_projection():
    model = _small_model()
    parameters = ts.codebook_parameters(model)
    assert set(parameters) == {"level0", "level1", "level2"}
    for level, (name, parameter) in enumerate(parameters.items()):
        raw = [value for key, value in model.rq.vq_layers[level].named_parameters()
               if key.startswith("embed")]
        assert any(parameter is value for value in raw), f"{name} is not a raw quantiser parameter"
