"""Issue #72 — the straight-through estimator makes the L2+L3 encoder gradient identically zero.

Establishes the facts that decide R5-10's fate, using a real model, a real forward and real
``torch.autograd.grad`` calls. No source-string assertions and no flag checks.

What is proved here
-------------------
1. ``‖d residual_k / d encoder‖ == 0`` for every k >= 1, with the cause traced to the
   straight-through line in ``model/layers.py``.
2. The zero is STRUCTURAL, not an R5-10 artefact: it is byte-identical in the pinned Resample
   parent, so no arm ever trained has received an L2+L3 encoder gradient.
3. The counterfactual: rebuilding the residual chain from the quantizer's ``embed_code`` (i.e. the
   pre-straight-through vector) restores a non-zero gradient. That is the concrete fix, and it is
   NOT applied here because it would change the pinned baseline every ER threshold is measured
   against.
4. ``w_b / valid_levels`` is the baseline's own per-level mean, not a PCGrad-ism, so R4-08/R5
   preserve it exactly and it must not be "fixed" in isolation.

Run: python results/stage2_RQ-VAE/r45_gradient_integrity_audit/test_ste_residual_chain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K  # noqa: E402

T = K.load_arm("r5_10")
N_SRC = 24
IN_DIM = 32
SEED = 1234


def build(seed: int = SEED, n_src: int = N_SRC):
    torch.manual_seed(seed)
    model = T.RQVAE(T._tokenizer_config(), in_dim=IN_DIM)
    emb = torch.randn(2 * n_src, IN_DIM, generator=torch.Generator().manual_seed(seed + 1))
    return model, emb, torch.arange(n_src), (torch.arange(n_src) + 1) % n_src


def encoder_l1(model, emb):
    return list(model.encoder.parameters())


def grad_l1(vectors, encoder) -> float:
    total = 0.0
    for vec in vectors:
        grads = torch.autograd.grad(vec, encoder, retain_graph=True, allow_unused=True)
        total += sum(float(g.abs().sum()) for g in grads if g is not None)
    return total


def residual_chain(model, emb, *, straight_through: bool):
    """The residual trace exactly as ``train_rqvae`` builds it, with a switchable STE.

    ``straight_through=True`` reproduces production: ``vq_layer(residual)`` returns
    ``x + (x_q - x).detach()``. ``straight_through=False`` rebuilds the same trace from the
    quantizer's own ``embed_code``, i.e. the pre-straight-through quantized vector.
    """
    residual = model.encoder(emb)
    traces = [residual]
    for layer in model.rq.vq_layers:
        quant, _, _, embed_ind = layer(residual)
        if not straight_through:
            quant = F.embedding(embed_ind, layer.get_code_embs()).view(residual.shape)
        residual = residual - quant
        traces.append(residual)
    return torch.stack(traces[: model.rq.codebook_num], dim=0)


def behaviour_split(model, emb, source_ids, target_ids, residuals):
    l1_loss, other_loss, valid = T.behavior_contrastive_components(
        residuals, source_ids, target_ids, None)
    return l1_loss, other_loss, valid


def test_production_residual_chain_has_zero_encoder_gradient_beyond_level_0():
    """d residual_k / d encoder must be exactly 0 for k >= 1 under the production STE."""
    model, emb, src, tgt = build()
    encoder = encoder_l1(model, emb)
    residuals = residual_chain(model, emb, straight_through=True)
    assert residuals.shape[0] == 3, residuals.shape
    g0 = grad_l1([residuals[0].sum()], encoder)
    g1 = grad_l1([residuals[1].sum()], encoder)
    g2 = grad_l1([residuals[2].sum()], encoder)
    assert g0 > 0.0, f"level 0 must reach the encoder, got {g0}"
    assert g1 == 0.0, f"level 1 must have an exactly zero encoder gradient, got {g1}"
    assert g2 == 0.0, f"level 2 must have an exactly zero encoder gradient, got {g2}"
    print(f"  production STE: |d residual[0]/d enc|_1 = {g0:.6e}, "
          f"level1 = {g1:.3e}, level2 = {g2:.3e}", flush=True)


def test_ste_is_the_cause():
    """The same residual trace built WITHOUT the straight-through term is differentiable."""
    model, emb, src, tgt = build()
    encoder = encoder_l1(model, emb)
    residuals = residual_chain(model, emb, straight_through=False)
    g0 = grad_l1([residuals[0].sum()], encoder)
    g1 = grad_l1([residuals[1].sum()], encoder)
    g2 = grad_l1([residuals[2].sum()], encoder)
    assert g1 > 0.0 and g2 > 0.0, (
        f"rebuilding the trace from embed_code must restore the chain gradient, got "
        f"level1 = {g1:.6e}, level2 = {g2:.6e}")
    print(f"  embed_code trace: level1 = {g1:.6e}, level2 = {g2:.6e} "
          f"(non-zero: the STE was the whole cause)", flush=True)


def test_behaviour_split_gradient_is_zero_for_the_l2_l3_term():
    """The encoder gradient of the L2+L3 behaviour term is identically zero in production."""
    model, emb, src, tgt = build()
    encoder = encoder_l1(model, emb)
    residuals = residual_chain(model, emb, straight_through=True)
    l1_loss, other_loss, valid = behaviour_split(model, emb, src, tgt, residuals)
    assert valid == 3, valid
    assert float(other_loss.detach()) > 0.0, "the L2+L3 loss VALUE must be non-trivial"
    g_l1 = T._r4_08_global_average(T._r4_08_encoder_gradients(l1_loss, encoder, retain=True))
    g_other = T._r4_08_global_average(
        T._r4_08_encoder_gradients(other_loss, encoder, retain=True))
    l1_norm = float(torch.cat([g.reshape(-1) for g in g_l1]).norm())
    other_norm = float(torch.cat([g.reshape(-1) for g in g_other]).norm())
    assert l1_norm > 0.0, "the L1 term must reach the encoder"
    assert other_norm == 0.0, (
        f"the L2+L3 term's encoder gradient must be exactly zero, got {other_norm}")
    print(f"  ||g_L1|| = {l1_norm:.6e}, ||g_other|| = {other_norm:.3e} "
          f"(other_loss value = {float(other_loss.detach()):.4f})", flush=True)


def test_w_b_over_valid_levels_is_the_baseline_per_level_mean():
    """w_b/valid_levels is the baseline loss's own per-level mean, so R4-08/R5 must preserve it."""
    model, emb, src, tgt = build()
    residuals = residual_chain(model, emb, straight_through=True)
    l1_loss, other_loss, valid = behaviour_split(model, emb, src, tgt, residuals)
    baseline_loss = T.behavior_contrastive_loss(residuals, src, tgt, None)
    split = (l1_loss + other_loss) / max(valid, 1)
    rel = float((split - baseline_loss).abs().max()) / (float(baseline_loss.abs()) + 1e-12)
    assert rel < 1e-6, (
        f"(l1 + other)/valid_levels must reproduce behavior_contrastive_loss exactly, rel={rel}")
    assert abs(float(split) - float(baseline_loss)) <= 1e-6 * max(
        abs(float(baseline_loss)), 1.0), (
        f"the split mean {float(split):.8f} and the baseline loss {float(baseline_loss):.8f} "
        "must be numerically identical: w_b/valid_levels is the baseline's own per-level mean, "
        "not something PCGrad introduced")
    print(f"  behavior_contrastive_loss = {float(baseline_loss):.8f}, "
          f"(l1+other)/{valid} = {float(split):.8f} (identical)", flush=True)


def test_structural_zero_holds_on_the_pinned_baseline_tree_too():
    """The property is inherited from the pinned Resample parent, not introduced by R4/R5."""
    baseline_tree = REPO / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/model/layers.py"
    arm_tree = REPO / "stage2_RQ-VAE/TIGER_r5_10/model/layers.py"
    assert baseline_tree.read_bytes() == arm_tree.read_bytes(), (
        "model/layers.py is byte-identical between the pinned baseline and r5_10, so the "
        "straight-through line cannot be an R4/R5 regression")
    print("  model/layers.py is byte-identical between TIGER_BEHAVIOR_SAMPLE_RQ-VAE and "
          "TIGER_r5_10", flush=True)


def test_r5_10_projection_against_a_zero_reference_is_a_passthrough():
    """With a zero reference the projection is provably a no-op, which is R5-10's actual run."""
    g_l1 = torch.tensor([1.0, -2.0, 0.5])
    zero = torch.zeros_like(g_l1)
    out = T.r4_project_conflict(g_l1.clone(), zero)
    assert torch.equal(out, g_l1), "projecting against zero must pass g_L1 through unchanged"
    before = T.R4_08_PROJECTION_TRACE["active_count"]
    T.r4_project_conflict(g_l1.clone(), zero)
    assert T.R4_08_PROJECTION_TRACE["active_count"] == before, (
        "the zero-reference branch must not even count as an active projection")
    print("  r4_project_conflict(g_L1, 0) is a bitwise passthrough and fires no projection",
          flush=True)


def main() -> int:
    tests = [
        test_production_residual_chain_has_zero_encoder_gradient_beyond_level_0,
        test_ste_is_the_cause,
        test_behaviour_split_gradient_is_zero_for_the_l2_l3_term,
        test_w_b_over_valid_levels_is_the_baseline_per_level_mean,
        test_structural_zero_holds_on_the_pinned_baseline_tree_too,
        test_r5_10_projection_against_a_zero_reference_is_a_passthrough,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}\n     {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}\n     {type(exc).__name__}: {exc}", flush=True)
        else:
            print(f"pass {test.__name__}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed", flush=True)
    if failed:
        print("STE_TESTS_RED", flush=True)
        return 1
    print("ALL TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
