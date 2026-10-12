from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_rqvae as tr


def close(a, b, atol=1e-6):
    torch.testing.assert_close(a, b, atol=atol, rtol=1e-6)


def reference_a(residuals, source_ids, target_ids, successor_indicator=None):
    batch = len(source_ids)
    levels = []
    for level in residuals:
        logits = -torch.cdist(level[:batch], level[batch:], p=2) / tr.BEHAVIOR_TEMPERATURE
        mask = target_ids[:, None].eq(target_ids[None, :])
        mask.fill_diagonal_(False)
        if successor_indicator is not None:
            blocked = torch.from_numpy(np.asarray(successor_indicator[source_ids.numpy()][:, target_ids.numpy()].todense())).bool()
            blocked.fill_diagonal_(False)
            mask |= blocked
        logits = logits.masked_fill(mask, -torch.inf)
        valid = source_ids.ne(target_ids)
        if valid.any():
            labels = torch.arange(batch)[valid]
            levels.append(F.cross_entropy(logits[valid], labels))
    return torch.stack(levels).mean() if levels else residuals.sum() * 0.0


def directional_logits(residual, heads, symmetric):
    b = residual.shape[0] // 2
    src, cand = residual[:b], residual[b:]
    fwd = heads.project_source(src) @ heads.project_target(cand).T / tr.BEHAVIOR_TEMPERATURE
    if not symmetric:
        return fwd
    rev = heads.project_source(cand) @ heads.project_target(src).T / tr.BEHAVIOR_TEMPERATURE
    return 0.5 * (fwd + rev)


def main():
    torch.manual_seed(17)
    b, nlevel = 5, 3
    residuals = torch.randn(nlevel, 2 * b, tr.CODEBOOK_DIM, requires_grad=True)
    sources = torch.tensor([1, 2, 3, 4, 5])
    targets = torch.tensor([11, 12, 13, 14, 15])

    # T1 and T2: pin the original cdist objective and pair score symmetry.
    for indicator in (None,):
        close(tr.behavior_contrastive_loss(residuals, sources, targets, indicator), reference_a(residuals, sources, targets, indicator))
    from scipy.sparse import csr_matrix
    successor = csr_matrix((np.ones(2, dtype=np.int8), ([1, 2], [14, 15])), shape=(20, 20))
    close(tr.behavior_contrastive_loss(residuals, sources, targets, successor), reference_a(residuals, sources, targets, successor))
    ra, rb = residuals[0, 0], residuals[0, b]
    close(-torch.linalg.vector_norm(ra-rb) / tr.BEHAVIOR_TEMPERATURE, -torch.linalg.vector_norm(rb-ra) / tr.BEHAVIOR_TEMPERATURE)

    heads = tr.DirectionalHeads()
    # T3-T5: pair symmetry/direction and hand-computed CE.
    pair = torch.randn(2, tr.CODEBOOK_DIM)
    p_ab = torch.stack((pair[0], pair[1]))
    p_ba = torch.stack((pair[1], pair[0]))
    ab = directional_logits(p_ab, heads, True)[0, 0]
    ba = directional_logits(p_ba, heads, True)[0, 0]
    close(ab, ba)
    d_ab = directional_logits(p_ab, heads, False)[0, 0]
    d_ba = directional_logits(p_ba, heads, False)[0, 0]
    magnitude = float((d_ab-d_ba).detach().abs())
    assert magnitude > 1e-6, magnitude

    # Loss labels correspond to the same-index positive target column.
    for symmetric in (False, True):
        logits = torch.stack([directional_logits(level, heads, symmetric) for level in residuals])
        mask = targets[:, None].eq(targets[None, :]); mask.fill_diagonal_(False)
        valid = sources.ne(targets)
        expected = torch.stack([F.cross_entropy(x.masked_fill(mask, -torch.inf)[valid], torch.arange(b)[valid]) for x in logits]).mean()
        actual = tr.directional_residual_loss(residuals, sources, targets, heads, symmetric)
        close(actual, expected)

    # T6: aligned basis vectors choose matching candidate; swapping halves chooses corresponding swapped index.
    h = tr.DirectionalHeads()
    with torch.no_grad():
        for layer in (h.f_source, h.f_target):
            layer.weight.copy_(torch.eye(tr.CODEBOOK_DIM)); layer.bias.zero_()
    basis = torch.eye(tr.CODEBOOK_DIM)[:4]
    aligned = torch.cat((basis, basis * 2), dim=0)
    scores = directional_logits(aligned, h, False)
    assert torch.equal(scores.argmax(dim=1), torch.arange(4))
    swapped = torch.cat((aligned[4:], aligned[:4]), dim=0)
    assert torch.equal(directional_logits(swapped, h, False).argmax(dim=1), torch.arange(4))

    # T7/T8: exact duplicate target masking and optional false-negative masking.
    sid = torch.tensor([1, 2, 3, 4]); tid = torch.tensor([8, 8, 9, 10])
    duplicate = tid[:, None].eq(tid[None, :]); duplicate.fill_diagonal_(False)
    assert not duplicate.diagonal().any() and duplicate[0, 1] and duplicate[1, 0]
    valid = sid.ne(tid)
    raw = torch.zeros(1, 8, tr.CODEBOOK_DIM)
    manual = F.cross_entropy((-torch.cdist(raw[0, :4], raw[0, 4:]) / tr.BEHAVIOR_TEMPERATURE).masked_fill(duplicate, -torch.inf)[valid], torch.arange(4)[valid])
    close(tr.behavior_contrastive_loss(raw, sid, tid), manual)
    known = csr_matrix((np.ones(1, dtype=np.int8), ([2], [9])), shape=(12, 12))
    loss_without_successor_mask = tr.behavior_contrastive_loss(raw, sid, tid, None)
    assert torch.equal(loss_without_successor_mask, tr.behavior_contrastive_loss(raw, sid, tid, None))
    assert not duplicate[1, 2]
    blocked = known[sid.numpy()][:, tid.numpy()].toarray().astype(bool)
    assert blocked[1, 2]
    loss_with_successor_mask = tr.behavior_contrastive_loss(raw, sid, tid, known)
    close(loss_with_successor_mask, reference_a(raw, sid, tid, known))
    assert not torch.allclose(loss_without_successor_mask, loss_with_successor_mask)

    # T9: encoder residual inputs and each head receive finite, nonzero gradients.
    for symmetric in (False, True):
        enc = torch.nn.Linear(tr.CODEBOOK_DIM, tr.CODEBOOK_DIM)
        encoded = torch.stack([enc(x) for x in residuals.detach()]).requires_grad_(True)
        loss = tr.directional_residual_loss(encoded, sources, targets, heads, symmetric)
        grads = torch.autograd.grad(loss, tuple(enc.parameters()) + tuple(heads.parameters()), allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads[:2]) > 0
        assert all(float(g.abs().sum()) > 0 for g in grads[2:])

    def initialized(mode):
        torch.manual_seed(1234)
        np.random.seed(1234)
        model = tr.RQVAE(tr._tokenizer_config(), in_dim=tr.CODEBOOK_DIM)
        tr.initialize_tiger_weights(model)
        if mode != "none":
            model.directional_heads = tr.DirectionalHeads()
        model.init_codebook(torch.randn(512, tr.CODEBOOK_DIM))
        return {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith("directional_heads.")}
    none, sym, direc = initialized("none"), initialized("sym"), initialized("dir")
    assert none.keys() == sym.keys() == direc.keys()
    assert all(torch.equal(none[k], sym[k]) and torch.equal(none[k], direc[k]) for k in none)

    # T11: export is independent of attached heads.
    torch.manual_seed(8)
    model = tr.RQVAE(tr._tokenizer_config(), in_dim=tr.CODEBOOK_DIM)
    tr.initialize_tiger_weights(model)
    model.init_codebook(torch.randn(512, tr.CODEBOOK_DIM))
    emb = torch.randn(12, tr.CODEBOOK_DIM)
    base = model.get_indices(emb, infer_use_sk=False)
    model.directional_heads = tr.DirectionalHeads()
    with torch.no_grad():
        for p in model.directional_heads.parameters(): p.zero_()
    with_heads = model.get_indices(emb, infer_use_sk=False)
    del model.directional_heads
    after_delete = model.get_indices(emb, infer_use_sk=False)
    assert torch.equal(base, with_heads) and torch.equal(base, after_delete)

    expected_delta = 2 * (tr.CODEBOOK_DIM**2 + tr.CODEBOOK_DIM)
    model_a = tr.RQVAE(tr._tokenizer_config(), in_dim=tr.CODEBOOK_DIM)
    count_a = sum(p.numel() for p in model_a.parameters())
    heads_b = tr.DirectionalHeads()
    heads_c = tr.DirectionalHeads()
    count_b = count_a + heads_b.n_parameters()
    count_c = count_a + heads_c.n_parameters()
    assert heads_b.n_parameters() == heads_c.n_parameters() == expected_delta
    assert count_b == count_c and count_b - count_a == expected_delta
    x = torch.randn(9, tr.CODEBOOK_DIM)
    close(heads.project_source(x).norm(dim=-1), torch.ones(9))
    close(heads.project_target(x).norm(dim=-1), torch.ones(9))
    r = torch.randn(8, tr.CODEBOOK_DIM)
    l1 = directional_logits(r, heads, False)
    l2 = directional_logits(r * 13.0, heads, False)
    close(l1, l2)
    old_temp = tr.BEHAVIOR_TEMPERATURE
    try:
        tr.BEHAVIOR_TEMPERATURE = old_temp * 2
        l3 = directional_logits(r, heads, False)
    finally:
        tr.BEHAVIOR_TEMPERATURE = old_temp
    assert not torch.allclose(l1, l3)
    print(f"T1-T14 PASS; directional_score_delta={magnitude:.9g}; parameters_A={count_a}; parameters_B={count_b}; parameters_C={count_c}; parameter_delta={expected_delta}")


if __name__ == "__main__":
    main()
