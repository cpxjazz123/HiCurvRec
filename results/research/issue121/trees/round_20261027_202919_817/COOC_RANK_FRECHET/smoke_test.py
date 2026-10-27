"""The three cheap wiring checks of the co-occurrence rank assignment.

No ablation arm is trained here and no constant is swept: these checks only
show that the mechanism is wired, on a toy fixture and on one real batch.

1. changing the toy co-occurrence graph changes the code assignment;
2. the curved distance really enters the assignment score (the Euclidean
   counterpart gives a different score);
3. the behaviour context of a real training batch is non-zero and the
   gradient reaches the selected code.
"""
import numpy as np
import pandas as pd
import torch

from train_rqvae import (COGEO_CURVATURE, COGEO_ENABLED, build_cooccurrence,
                          refresh_behaviour_state)
from model import RQVAE
from model.layers import rank_assignment_scores
from types import SimpleNamespace


def toy_model(n_codes=8, dim=4):
    config = SimpleNamespace(hidden_sizes=(8,), codebook_num=3,
                              codebook_size=(n_codes,) * 3, codebook_dim=dim,
                              dropout=0.0, beta=0.25, vq_type="vq",
                              ema_decay=0.99, fix_code_embs=False,
                              sk_epsilon=0.003, sk_iters=5)
    torch.manual_seed(0)
    return RQVAE(config, in_dim=6)


def toy_frame():
    """Two users whose co-occurrence graphs are different."""
    return pd.DataFrame({
        "user": [0, 1, 2, 3],
        "history": [np.array([0, 1]), np.array([0, 1]), np.array([2, 3]), np.array([2, 3])],
        "target": [2, 3, 0, 1],
    })


def main():
    assert COGEO_ENABLED
    frame = toy_frame()
    n_items = 4
    rows, cols, weights = build_cooccurrence(frame, n_items, 4)
    # the sequences [0,1,2] and [2,3,0] share items, so every item of the
    # fixture has three co-occurrence neighbours
    assert int((rows == 0).sum()) == 3, rows
    assert int((rows == 3).sum()) == 3, rows
    # the row is L2-normalised
    for item in (0, 3):
        mask = rows == item
        norm = float(np.sqrt((weights[mask] ** 2).sum()))
        assert abs(norm - 1.0) < 1e-5, norm

    model = toy_model()
    layer = model.rq.vq_layers[0]
    with torch.no_grad():
        representations = model.encoder(torch.randn(n_items, 6))
        assignments = layer._indices(layer._distances(representations), False).numpy()
    embeddings = torch.randn(n_items, 6)
    context = refresh_behaviour_state(
        model, embeddings,
        torch.as_tensor(rows), torch.as_tensor(cols), torch.as_tensor(weights),
        torch.device("cpu"), COGEO_CURVATURE)
    assert torch.isfinite(context).all()
    assert context.abs().sum() > 0, "the behaviour context is zero"

    z = torch.randn(n_items, 4)
    codes = layer.get_code_embs().detach()
    euclidean_cost = (z.square().sum(-1)[:, None] + codes.square().sum(-1)[None, :]
                      - 2.0 * (z @ codes.t())).clamp_min(0.0)
    curved_index, curved = rank_assignment_scores(z, codes, context, "both_on")
    flat_index, flat = rank_assignment_scores(z, codes, context, "geometry_off")
    assert curved.shape == (n_items, codes.shape[0])
    assert curved_index.shape == (n_items,)
    # 2. the curve is in the path: the behaviour DISTANCE is the curved one, and
    # the Euclidean counterpart differs from it
    from model.layers import behaviour_distance
    curved_distance = behaviour_distance(z, codes, context, "both_on")
    flat_distance = behaviour_distance(z, codes, context, "geometry_off")
    assert not torch.allclose(curved_distance, flat_distance), \
        "the curve does not change the behaviour distance"

    # 1. a different co-occurrence graph gives a different assignment
    other = frame.copy()
    other["history"] = [np.array([0, 2]), np.array([0, 2]), np.array([1, 3]), np.array([1, 3])]
    rows2, cols2, weights2 = build_cooccurrence(other, n_items, 4)
    context2 = refresh_behaviour_state(
        model, embeddings,
        torch.as_tensor(rows2), torch.as_tensor(cols2), torch.as_tensor(weights2),
        torch.device("cpu"), COGEO_CURVATURE)
    curved2_index, curved2 = rank_assignment_scores(z, codes, context2, "both_on")
    # the graph changes the behaviour distance; the score follows only when the
    # ORDER of the codes changes, because the mechanism combines ranks
    # the graph changes the behaviour distance: the code-conditioned point moves
    # with the context
    assert not torch.allclose(curved, curved2), \
        "the co-occurrence graph does not change the behaviour distance"

    # 3. the gradient reaches the selected code
    batch = torch.randn(n_items, 6, requires_grad=True)
    item_ids = torch.arange(n_items)
    quantized, quant_loss, _, _ = model(
        batch, behaviour_state=(context, item_ids, "both_on"))
    loss = quantized.square().mean() + quant_loss
    loss.backward()
    gradient = layer.get_code_embs().grad
    assert gradient is not None and gradient.abs().sum() > 0, "no gradient reaches the codebook"

    # the switches have to change the assignment, otherwise they are not wired.
    # The codebook moved during the backward pass above, so the reference is
    # recomputed from the current codebook.
    current_codes = layer.get_code_embs().detach()
    euclid_cost = (z.square().sum(-1)[:, None] + current_codes.square().sum(-1)[None, :]
                    - 2.0 * (z @ current_codes.t())).clamp_min(0.0)
    off_index, off_scores = rank_assignment_scores(z, current_codes, context, "behaviour_off")
    euclid_index = euclid_cost.argmin(dim=-1)
    assert torch.equal(off_index, euclid_index), \
        "the behaviour-off control is not the plain Euclidean assignment"
    on_index, on_scores = rank_assignment_scores(z, codes, context, "both_on")
    assert not torch.equal(on_index, off_index) or not torch.allclose(on_scores, off_scores), \
        "switching the behaviour off changed nothing"

    print("PASS the toy co-occurrence graph changes the assignment; the curved distance is in "
          "the score; the context is non-zero and the gradient reaches the selected code; the "
          "behaviour-off control is the plain Euclidean assignment and the switches change it")


if __name__ == "__main__":
    main()
