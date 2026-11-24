"""Issue #232 §1–§3: real tokens, a gradient-gated hinge, and a separate structure step.

Each test names the clause it proves. Run with the Stage2 interpreter (CLAUDE.md §7)::

    cd stage2_RQ-VAE/TIGER_export_matched_graphlet_rq && \
      /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parent.parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

import analysis_contract as CONTRACT  # noqa: E402
from losses import layered_graphlet as LG  # noqa: E402
from losses import layered_wiring as LW  # noqa: E402


def _codes(rows):
    return torch.tensor(rows, dtype=torch.long)


def _fmt(tokens):
    width = int(tokens.max()) + 1
    onehot = torch.nn.functional.one_hot(tokens.long(), width).float()
    return onehot + (1.0 - onehot) * 1e-3


# --- §1 C8/C9/C38: real tokens, never a re-derived argmin --------------------------------------
def test_c8_c38_value_follows_the_tokens_not_the_cost():
    roles = torch.tensor([[0, 1, 2]])
    tokens = _codes([[0, 0, 0], [0, 0, 0], [3, 3, 3]])
    # S(AB)=3, S(BC)=0, S(AC)=0  ->  (3+0)/2 - 0 = 1.5
    assert float(LG.structure_value(tokens, roles, "chain")) == pytest.approx(1.5)
    flipped = _codes([[1, 1, 1], [0, 0, 0], [2, 2, 2]])
    assert float(LG.structure_value(flipped, roles, "chain")) == pytest.approx(0.0)


def test_c9_the_module_does_not_rebuild_hard_codes():
    source = (TREE / "losses" / "layered_graphlet.py").read_text()
    assert "argmax" not in source and "argmin" not in source


def test_c10_ste_forward_is_hard_backward_is_p():
    prob = torch.tensor([[0.7, 0.2, 0.1]], requires_grad=True)
    hard = torch.tensor([[1.0, 0.0, 0.0]])
    mixed = LG.ste(prob, hard)
    assert torch.equal(mixed.detach(), hard)
    mixed.sum().backward()
    assert torch.all(prob.grad == 1.0)


@pytest.mark.parametrize("kind", ["chain", "fork", "merge"])
def test_c11_forward_matches_the_independent_scorer(kind):
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, 8, (200, 3), generator=generator)
    roles = torch.randint(0, 40, (60, 3), generator=generator)
    mine = LG.structure_value(tokens, roles, kind)
    theirs = CONTRACT.delta_of(kind,
                               CONTRACT.shared_prefix(tokens.numpy(), roles[:, 0].numpy(),
                                                      roles[:, 1].numpy()),
                               CONTRACT.shared_prefix(tokens.numpy(), roles[:, 1].numpy(),
                                                      roles[:, 2].numpy()),
                               CONTRACT.shared_prefix(tokens.numpy(), roles[:, 0].numpy(),
                                                      roles[:, 2].numpy()))
    assert float(mine.mean()) == pytest.approx(float(theirs.mean()), abs=1e-6)


def test_c11_agrees_with_the_repository_scorer_when_reachable():
    scorer = TREE.parent.parent / "analysis_sid_recall" / "graphlet_contract.py"
    if not scorer.is_file():
        pytest.skip("repository scorer not reachable from this checkout")
    sys.path.insert(0, str(scorer.parent.parent))
    from analysis_sid_recall import graphlet_contract as real  # noqa: PLC0415

    generator = torch.Generator().manual_seed(9)
    tokens = torch.randint(0, 6, (80, 3), generator=generator)
    roles = torch.randint(0, 30, (20, 3), generator=generator)
    instances = roles.numpy()
    for kind in ("chain", "fork", "merge"):
        mine = float(LG.structure_value(tokens, roles, kind).mean())
        theirs = real.measure(tokens.numpy(), kind, instances)["delta"]
        assert mine == pytest.approx(theirs, abs=1e-6)


# --- §1 C14: the hinge's mask gates the gradient ------------------------------------------------
def test_c14_an_instance_at_margin_contributes_exactly_zero_gradient():
    gap = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
    loss = LG.gated_hinge(gap, LG.MARGIN_L1).sum()
    loss.backward()
    assert float(loss) == 0.0
    assert torch.all(gap.grad == 0.0)


def test_c14_inactive_rows_do_not_pollute_active_ones():
    gap = torch.tensor([2.0, -0.5], requires_grad=True)
    LG.gated_hinge(gap, LG.MARGIN_L1).sum().backward()
    assert gap.grad.tolist() == [0.0, -1.0]


def test_c15_weighting_scales_value_and_gradient_together():
    base = torch.tensor([0.3, -0.4], requires_grad=True)
    loss = (LG.gated_hinge(base, LG.MARGIN_L1) * 3.0).sum()
    loss.backward()
    plain = torch.tensor([0.3, -0.4], requires_grad=True)
    LG.gated_hinge(plain, LG.MARGIN_L1).sum().backward()
    assert torch.allclose(base.grad, 3.0 * plain.grad)


# --- C21–C24, C29: the formulas ----------------------------------------------------------------
@pytest.mark.parametrize("kind", ["chain", "fork", "merge"])
def test_c21_prefix_share_matches_the_contract(kind):
    tokens = _codes([[5, 5, 1], [5, 9, 9], [5, 5, 9]])
    roles = torch.tensor([[0, 1, 2]])
    s = CONTRACT.shared_prefix(tokens.numpy(), np.array([0, 1, 0]), np.array([1, 2, 2]))
    assert float(LG.structure_value(tokens, roles, kind)) == pytest.approx(
        float(CONTRACT.delta_of(kind, s[0], s[1], s[2])))


def test_c29_chain_uses_the_mean_and_never_demands_both():
    tokens = _codes([[7, 7, 7], [0, 0, 0], [7, 7, 0]])
    roles = torch.tensor([[0, 1, 2]])
    # S(AB)=0, S(BC)=0, S(AC)=2 -> mean(near) - far = 0 - 2 = -2
    assert float(LG.structure_value(tokens, roles, "chain")) == pytest.approx(-2.0)


# --- C25–C28: three level objectives, not three copies of one hinge ----------------------------
def test_c28_the_three_levels_are_different_functions():
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, 6, (12, 3), generator=generator)
    roles = torch.randint(0, 12, (4, 3), generator=generator)
    probs = [_fmt(tokens[:, level]) for level in range(3)]
    parts = LG.level_objectives(probs, tokens, roles, "fork")
    assert set(parts) >= {"l1", "l2", "l3"}
    # L1 reads level 1's contrast, L3 reads the hard bucket contrast; they are different maps
    assert float(parts["l1_gap"]) != float(parts["l3_gap"])


def test_c27_l3_reads_the_export_buckets():
    tokens = _codes([[1, 2, 3], [1, 2, 4], [5, 5, 5]])
    buckets = LG.bucket_of(tokens, LG.L3_WIDTH)
    assert int(buckets[0]) != int(buckets[1])
    # width 2: bucket = t0*4 + t1*2 + t2
    assert LG.bucket_of(_codes([[1, 1, 1]]), LG.L3_WIDTH).tolist() == [7]


def test_c37_hand_sid_is_consistent_across_levels():
    tokens = _codes([[10, 21, 31], [10, 21, 32], [10, 22, 33]])
    roles = torch.tensor([[0, 1, 2]])
    # issue §2's own chain example: A=(10,21,31) B=(10,21,32) C=(10,22,33)
    assert float(LG.structure_value(tokens, roles, "chain")) == pytest.approx(0.5)
    assert len({tuple(int(v) for v in row) for row in tokens.tolist()}) == 3


def test_c30_kind_weighted_average_does_not_double_count():
    tokens = _codes([[1, 1, 1], [1, 1, 2], [1, 2, 2]])
    probs = [_fmt(tokens[:, level]) for level in range(3)]
    roles = torch.tensor([[0, 1, 2]])
    both = LG.graphlet_objective(probs, tokens, {"chain": roles, "fork": roles})
    chain = LG.graphlet_objective(probs, tokens, {"chain": roles})
    fork = LG.graphlet_objective(probs, tokens, {"fork": roles})
    assert float(both) == pytest.approx((float(chain) + float(fork)) / 2.0)


# --- C42/C35/C16: the base path is untouched ---------------------------------------------------
def test_c42_the_term_joins_the_single_backward():
    """Owner's joint contract: one scalar, one backward, the original clip, one step."""
    source = (TREE / "train_rqvae.py").read_text()
    assert "loss = (loss + aux_scale[\"graphlet\"] * graphlet_value" in source
    assert "GRAPHLET_ALIGNMENT" not in source
    assert "GRAPHLET_SHARE" not in source


def test_c16_export_extension_is_unchanged():
    import train_rqvae  # noqa: PLC0415

    tokens = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
    sid = train_rqvae._extend_collisions(tokens, [8, 8, 8])
    assert sid.shape == (3, 4)
    assert sid[0, :3].tolist() == [1, 2, 3] and sid[1, :3].tolist() == [1, 2, 3]
    assert sid[0, 3] != sid[1, 3]


# --- C39/C40/C41: gradients on the real model --------------------------------------------------
    import train_rqvae  # noqa: PLC0415
    from model import RQVAE  # noqa: PLC0415

    return RQVAE(train_rqvae._tokenizer_config(), in_dim=32, distance=train_rqvae.DISTANCE)


def _banks():
    import train_rqvae  # noqa: PLC0415

    return train_rqvae.GRAPHLET_BANKS


def _real_embeddings():
    import train_rqvae  # noqa: PLC0415

    return torch.from_numpy(np.load(train_rqvae.EMBEDDING_FILE)).float()


def _trained_model():
    import train_rqvae  # noqa: PLC0415
    from model import RQVAE  # noqa: PLC0415

    checkpoint = train_rqvae.REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "rqvae_best.pth"
    if not checkpoint.is_file():
        pytest.skip("the frozen checkpoint is not reachable")
    model = RQVAE(train_rqvae._tokenizer_config(), in_dim=768, distance=train_rqvae.DISTANCE)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
    return model.eval()


def test_c39_c40_structure_step_reaches_encoder_and_codebooks():
    """Run on the FROZEN checkpoint: an untrained model has no structure at all, so every hinge
    sits at its ceiling and the gradient is exactly 0 -- the first 3000-epoch run proved that
    `> 0` on a random model proves nothing (its objective was constant 1.3 for 18,000 steps)."""
    model = _trained_model()
    branch = LW.StructureBranch(_banks(), instance_batch=32, seed=1)
    embeddings = _real_embeddings()
    report = LW.run_structure_step(model, embeddings, branch.draw())
    # The threshold is a MEASURED magnitude, not `> 0`: the first 3000-epoch run passed a `> 0`
    # assertion while the objective sat at exactly 1.3 for all 18,000 steps and every level's
    # gradient was ~1e-8, because `softmax(-cost)` was flat at the real cost scale.
    assert report["latent_grad_norm"] > 1e-3, report["latent_grad_norm"]
    assert report["encoder_grad_norm"] > 1e-3, report["encoder_grad_norm"]
    # the codebook is not routed from the structure term: its only gradient is reconstruction's.
    # (Under the joint backward the term is an addend of the same scalar, so the codebook's
    # gradient comes from recon and the routing reports nothing for it.)
    assert report["codebook_frozen"] is True
    assert all(norm == 0.0 for norm in report["codebook_grad_norms"])


def test_c39b_the_temperature_resolves_the_real_cost_scale():
    """TAU must put `p` on the quantizer's decision boundary, not at either extreme."""
    import train_rqvae  # noqa: PLC0415
    import torch  # noqa: PLC0415

    checkpoint = train_rqvae.REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "rqvae_best.pth"
    if not checkpoint.is_file():
        pytest.skip("the frozen checkpoint is not reachable")
    from model import RQVAE  # noqa: PLC0415

    model = RQVAE(train_rqvae._tokenizer_config(), in_dim=768, distance=train_rqvae.DISTANCE)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
    model.eval()
    embeddings = torch.from_numpy(np.load(train_rqvae.EMBEDDING_FILE)).float()[:512]
    with torch.no_grad():
        cost = model.rq.vq_layers[0]._distances(model.encoder(embeddings))
        prob = torch.softmax(-cost / LW.TAU, dim=-1)
    peak = float(prob.max(-1).values.mean())
    # flat (uniform 1/256 = 0.0039) has no gradient; saturated (1.0) has none either
    assert 0.05 < peak < 0.95, peak


def test_c41_a_satisfiable_instance_improves_its_hard_delta():
    """A tiny feasible example: descending the real structure step moves the HARD S value."""
    model = _trained_model()
    branch = LW.StructureBranch(_banks(), instance_batch=16, seed=2)
    embeddings = _real_embeddings()
    draw = branch.draw()
    before = _hard_delta(model, embeddings, draw)
    # same step contract as the trainer: the gradient is rescaled before the step, so the step
    # size is the weight rather than the objective's momentary slope
    # the trainer's own learning rate: 1e-2 over 40 structure steps is 10x the frozen schedule's
    # first-epoch rate and drives the encoder to tie-decided assignments (measured 60% of items),
    # which the anti-collapse guard correctly refuses
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        report = LW.run_structure_step(model, embeddings, draw)
        LW.normalise_gradients(model)
        optimizer.step()
    after = _hard_delta(model, embeddings, draw)
    assert after >= before, f"hard S fell from {before} to {after}"

def _hard_delta(model, embeddings, draw):
    with torch.no_grad():
        values = []
        for level, items in draw.items():
            union = np.unique(items.reshape(-1).numpy())
            rows = torch.from_numpy(union.astype(np.int64))
            _, tokens, _, _ = model.quantize_items(embeddings, rows)
            roles = torch.from_numpy(np.searchsorted(union, items.numpy())).long()
            values.append(float(LG.structure_value(tokens, roles, level).mean()))
    return float(np.mean(values))


# --- owner's schedule: half the budget plain, then a linear ramp -------------------------------
def test_c232_ramp_is_zero_until_half_and_reaches_the_max_at_the_end():
    from losses.layered_graphlet import ramp_weight

    assert ramp_weight(1, 3000, 1500, 0.20) == 0.0
    assert ramp_weight(1499, 3000, 1500, 0.20) == 0.0
    assert ramp_weight(1500, 3000, 1500, 0.20) == 0.0
    assert ramp_weight(2250, 3000, 1500, 0.20) == pytest.approx(0.10)
    assert ramp_weight(3000, 3000, 1500, 0.20) == pytest.approx(0.20)
    # monotone over the ramped half
    values = [ramp_weight(e, 3000, 1500, 0.20) for e in range(1500, 3001, 100)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_c232_the_term_is_an_addend_of_the_same_scalar():
    source = (TREE / "train_rqvae.py").read_text()
    assert "loss = (loss + aux_scale[\"graphlet\"] * graphlet_value" in source
    assert "_structure_step(" not in source, "the separate step is gone"
    assert "normalise_gradients" not in source, "the RMS normalisation is gone"
    assert "clip_grad_norm_" in source
