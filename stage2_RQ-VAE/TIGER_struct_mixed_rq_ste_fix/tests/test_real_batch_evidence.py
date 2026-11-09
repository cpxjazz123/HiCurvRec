"""Evidence on the real TRAIN table: gradients and one real optimizer step.

No stubs and no skips: the parquet and the embedding file are in the repository.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import native_product
import structure as S
import train_rqvae as T
import training_hooks as H
from model.model import RQVAE


def _setup():
    torch.manual_seed(0)
    np.random.seed(0)
    emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
    items = torch.from_numpy(emb).float()
    model = RQVAE(T._tokenizer_config(), in_dim=emb.shape[1])
    if T.XAVIER_INIT:
        T.initialize_tiger_weights(model)
    frame = pd.read_parquet(T.TRAIN_FILE)
    ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    with torch.no_grad():
        model.init_codebook(items[torch.from_numpy(ids[:2048])])
    batch = items[torch.from_numpy(ids[:T.BATCH_SIZE_PER_RANK])]
    return model, items, batch


def _grad_norms(term, params):
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return [0.0 if g is None else float(g.abs().sum()) for g in grads]


def test_the_real_batch_gives_encoder_and_codebook_gradients():
    model, items, batch = _setup()
    recon, quant_loss, _, _ = model.forward(batch)
    loss = model.compute_loss(batch, recon, quant_loss)[0]
    codebooks = [layer.get_code_embs() for layer in model.rq.vq_layers]
    encoder = [p for p in model.encoder.parameters()]
    enc = _grad_norms(loss, encoder)
    cbs = _grad_norms(loss, codebooks)
    assert sum(enc) > 0, "reconstruction gives the encoder no gradient"
    assert all(value > 0 for value in cbs), cbs


def test_the_structure_term_reaches_both_blocks_and_one_step_moves_the_codes():
    model, items, batch = _setup()
    graph = S.build_graph(T.TRAIN_FILE, seed=T.SEED)
    recon, quant_loss, _, _ = model.forward(batch)
    base = model.compute_loss(batch, recon, quant_loss)[0]
    scale = H.calibrate_structure_scale(model, items, graph,
                                        __import__("random").Random(T.SEED), size=64)
    term = H.structure_term(model, items, graph, __import__("random").Random(T.SEED),
                            scale=scale, size=64)
    codebooks = [layer.get_code_embs() for layer in model.rq.vq_layers]
    encoder = [p for p in model.encoder.parameters()]
    assert sum(_grad_norms(term, encoder)) > 0
    cbs = _grad_norms(term, codebooks)
    assert all(value > 0 for value in cbs), cbs
    before = [p.detach().clone() for p in codebooks]
    opt = torch.optim.AdamW(model.parameters(), lr=float(T.LR))
    (base + term).backward()
    opt.step()
    moved = [float((b - p.detach()).abs().sum()) for b, p in zip(before, codebooks)]
    assert all(value > 0 for value in moved), moved


def test_two_items_with_one_sid_have_no_structure_gradient_between_them():
    import mixed_geometry as G
    point = torch.zeros(2, 32)
    point[:, 16] = 0.2
    point.requires_grad_(True)
    distance = G.mixed_sq(point[0], point[1], True)
    assert float(distance.detach()) == 0.0
    grad = torch.autograd.grad(distance, point)[0]
    assert float(grad.abs().sum()) == 0.0


if __name__ == "__main__":
    test_the_real_batch_gives_encoder_and_codebook_gradients()
    test_the_structure_term_reaches_both_blocks_and_one_step_moves_the_codes()
    test_two_items_with_one_sid_have_no_structure_gradient_between_them()
    print("ok")
