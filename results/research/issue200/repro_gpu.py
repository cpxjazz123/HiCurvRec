"""Reproduce the smoke's device-side assert with blocking CUDA launches, so the failing kernel names itself."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TREE = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_f0b_rq"


def main() -> int:
    sys.path.insert(0, str(TREE))
    spec = importlib.util.spec_from_file_location("f0b", TREE / "train_rqvae.py")
    arm = importlib.util.module_from_spec(spec)
    sys.modules["f0b"] = arm
    spec.loader.exec_module(arm)
    from model import RQVAE, sd_step, struct_distill

    device = torch.device("cuda:0")
    print("device", device, flush=True)
    embeddings = arm.load_embeddings(arm.EMBEDDING_FILE)
    frame = arm.pd.read_parquet(arm.TRAIN_FILE, columns=["seen_history", "target"])
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    edges = sorted({(int(h[-1]), int(t)) for h, t in
                    zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64))
                    if h is not None and len(h) > 0})
    neighbours: dict[int, list[int]] = {}
    for i, j in edges:
        neighbours.setdefault(i, []).append(j)
        neighbours.setdefault(j, []).append(i)
    two_hop = {node: struct_distill.two_hop_of(node, neighbours) for node in list(neighbours)[:20000]}
    n_items = int(max(max(e) for e in edges)) + 1
    print("items", len(embeddings), "graph n_items", n_items, "train_ids max", int(train_ids.max()), flush=True)

    model = RQVAE(arm._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    arm.initialize_tiger_weights(model)
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    all_emb = torch.from_numpy(np.ascontiguousarray(embeddings)).float()
    with torch.no_grad():
        model.init_codebook(all_emb[torch.from_numpy(train_ids)].to(device))
    print("codebook init ok", flush=True)

    model.eval()
    batch_ids = torch.from_numpy(train_ids[:1024])
    batch = all_emb[batch_ids].to(device)
    print("forward", flush=True)
    z = model.encoder(batch)
    readout, qloss, unused, tokens = model.rq(z)
    print("rq ok", tuple(readout.shape), flush=True)
    anchors = batch_ids[: sd_step.ANCHORS_PER_STEP]
    print("structural_loss", flush=True)
    loss = sd_step.structural_loss(model, all_emb.to(device), anchors, None, neighbours, two_hop, n_items)
    print("structural loss", float(loss.detach()), flush=True)
    loss.backward()
    print("backward ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
