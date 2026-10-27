"""Fast CPU smoke test for the coupled first-level assignment."""
import numpy as np
import pandas as pd
import torch
from train_rqvae import CoGeoAssignment, _repository_root, _extend_collisions
from model import RQVAE
from types import SimpleNamespace

def reference_counts(frame, n_items):
    """The original dictionary construction, kept as the equivalence check.

    The numpy construction of the co-consumption and the transition counts has
    to agree with it value by value, not only in the number of edges: a
    vectorised count that drops the multiplicities is a different behaviour
    signal, and a reviewer has to be able to see that it is not.
    """
    from collections import defaultdict
    rows = [defaultdict(float) for _ in range(n_items)]
    transitions = defaultdict(float)
    longest = {}
    users = frame["user"] if "user" in frame.columns else range(len(frame))
    for user, hist, target in zip(users, frame["history"], frame["target"]):
        seq = [int(x) for x in np.asarray(hist).reshape(-1)] + [int(target)]
        seq = [x for x in seq if 0 <= x < n_items]
        current = longest.get(int(user))
        if current is None or len(seq) > len(current):
            longest[int(user)] = seq
    for seq in longest.values():
        for p, i in enumerate(seq):
            for j in seq[p + 1:]:
                if i != j:
                    rows[i][j] += 1.0
                    rows[j][i] += 1.0
            if p:
                transitions[seq[p - 1], i] += 1.0
    return rows, transitions


def main():
    assert (_repository_root() / "stage2_RQ-VAE").is_dir()
    frame = pd.DataFrame({"history": [np.array([], dtype=np.int64), np.array([0])],
                          "target": [0, 1]})
    coupling = CoGeoAssignment(frame, 4)
    cfg = SimpleNamespace(hidden_sizes=(8,), codebook_num=3, codebook_size=(4,4,4),
        codebook_dim=3, dropout=0.0, beta=.25, vq_type="vq", ema_decay=.99,
        fix_code_embs=False, sk_epsilon=.003, sk_iters=5)
    model = RQVAE(cfg, in_dim=5)
    layer = model.rq.vq_layers[0]
    with torch.no_grad():
        z = model.encoder(torch.randn(4, 5))
        coupling.refresh(layer._indices(layer._distances(z), False).numpy(), 4, torch.device("cpu"))
    layer.region_assignment = coupling
    layer.region_item_ids = torch.tensor([0,1,2,3])
    x = torch.randn(4, 5)
    recon, qloss, _, _ = model(x)
    loss = recon.square().mean() + qloss
    assert torch.isfinite(loss)
    loss.backward()
    grad = next(m.weight.grad for m in model.encoder.modules() if isinstance(m, torch.nn.Linear))
    assert grad is not None and torch.isfinite(grad).all()
    sid = _extend_collisions(model.get_indices(x).numpy(), [4,4,4])
    assert sid.shape == (4, 4) and len(np.unique(sid, axis=0)) == 4
    # A deterministic fixture that proves the switches are sensitive: the
    # region profiles and the directed radar are set so that the coupled rule
    # cannot land on the same codes as the plain Euclidean choice. The same
    # batch, the same codebook, one reference (the Euclidean argmin) and three
    # variants, so the differences are the effect of the coupled score itself.
    frame2 = pd.DataFrame({"history": [np.array([0, 1]), np.array([0, 1]),
                                        np.array([2, 3]), np.array([2, 3])],
                            "target": [2, 3, 0, 1]})
    coupling2 = CoGeoAssignment(frame2, 4)
    # the vectorised counts must equal the dictionary counts value by value
    reference_rows, reference_transitions = reference_counts(frame2, 4)
    counted = {}
    for row, col, value in zip(coupling2.cooc_rows, coupling2.cooc_cols,
                               coupling2.cooc_counts):
        counted[(int(row), int(col))] = float(value)
    for item, row in enumerate(reference_rows):
        for neighbour, value in row.items():
            assert counted.get((item, neighbour), 0.0) == value, (item, neighbour)
    counted_transitions = {}
    for src, dst, value in zip(coupling2.trans_src, coupling2.trans_dst,
                               coupling2.trans_val):
        counted_transitions[(int(src), int(dst))] = float(value)
    assert counted_transitions == dict(reference_transitions), counted_transitions
    with torch.no_grad():
        coupling2.refresh(np.array([0, 1, 2, 3], dtype=np.int64), 4, torch.device("cpu"))
        # A codebook at the origin makes the geodesic term flat, so the three
        # variants are separated by the behaviour term and by the Euclidean
        # term instead of by a random draw.
        layer.get_code_embs().data.zero_()
        agree = torch.zeros((4, 4))
        for item, code in ((0, 1), (1, 1), (2, 3), (3, 3)):
            agree[item, code] = 0.1
        coupling2.agree_all = agree
        radar = torch.zeros((4, 4))
        radar[0, 3] = 1.0
        radar[1, 3] = 1.0
        radar[2, 1] = 1.0
        radar[3, 1] = 1.0
        coupling2.P_tensor = radar
        # the incoming transition mass the step reads
        coupling2.directed_all = radar
        layer.region_assignment = coupling2
        layer.region_item_ids = torch.tensor([0, 1, 2, 3])
        latent = model.encoder(x)
        coupling2.current_latent = latent
        # A fixed distance matrix, so the plain choice is the same every run.
        distances = torch.tensor([[0.01, 0.5, 0.5, 0.5],
                                  [0.5, 0.01, 0.5, 0.5],
                                  [0.5, 0.5, 0.01, 0.5],
                                  [0.5, 0.5, 0.5, 0.01]])
        variants = coupling2.compare_variants(layer.region_item_ids, distances,
                                               layer.get_code_embs())
    assert variants["items"] == 4
    assert variants["plain"] == [0, 1, 2, 3]
    # the behaviour term must move the assignment, otherwise it is a dead term
    assert variants["both_on"] > 0, "the coupled score did not change any code"
    assert variants["choices"]["both_on"] == [1, 1, 3, 3], variants["choices"]
    # both switches have to change the assignment: a switch that changes
    # nothing is a switch that is not wired.
    assert variants["choices"]["both_on"] != variants["choices"]["behaviour_off"], \
        "the behaviour switch changed nothing"
    assert variants["choices"]["both_on"] != variants["choices"]["geometry_off"], \
        "the geometry switch changed nothing"
    print("PASS forward/backward finite; encoder gradient; unique SID export shape=(4, 4)")
    print("VARIANTS", variants)

if __name__ == "__main__":
    main()
