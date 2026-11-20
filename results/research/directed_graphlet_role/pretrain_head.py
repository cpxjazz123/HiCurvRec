"""Issue #221 arm D4: pre-train the DIRECTED head on z, validate, save, freeze."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template"
sys.path.insert(0, str(TEMPLATE))

import drq  # noqa: E402
from directed_role import (DirectedTypeRoleHead, instance_targets, load_pool,  # noqa: E402
                           load_vocabulary)
from model import RQVAE  # noqa: E402
from train_config import load_train_config  # noqa: E402

REPO = HERE.parents[2]
STEPS = 3000
PER_FAMILY = 32
LR = 1e-3


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_train_config(TEMPLATE / "config" / "train_config.json")
    model_config = type("C", (), dict(hidden_sizes=tuple(config.HIDDEN_SIZES),
                                      codebook_num=config.CODEBOOK_NUM,
                                      codebook_size=tuple(config.CODEBOOK_SIZE),
                                      codebook_dim=config.CODEBOOK_DIM, dropout=0.0,
                                      beta=config.BETA, vq_type=config.VQ_TYPE,
                                      ema_decay=config.EMA_DECAY, fix_code_embs=False,
                                      sk_epsilon=config.SK_EPSILON, sk_iters=config.SK_ITERS))()
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                          "sentence_t5.npy").astype("float32"))
    model = RQVAE(model_config, in_dim=embeddings.shape[1]).to(device)
    baseline = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "rqvae_best.pth"
    model.load_state_dict(torch.load(baseline, map_location="cpu")["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    vocab = load_vocabulary()
    head = DirectedTypeRoleHead(dim=config.CODEBOOK_DIM, n_subtypes=vocab["n_subtypes"],
                                n_orbits=vocab["n_orbits"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    sampler, _vocab = drq.make_sampler("train", 42)
    diagnostics = {}
    for step in range(1, STEPS + 1):
        draw = sampler.sample()
        features = {}
        for family, (nodes, _orbits, _subtypes) in draw.items():
            count, width = nodes.shape
            index = nodes.reshape(-1).long()
            with torch.no_grad():
                latent = model.encoder(drq.gather(embeddings, index).to(device))
            features[family] = latent.view(count, width, -1)
        loss, diagnostics = drq.joint_loss(head, features, draw, vocab, drq.targets_for(draw, vocab))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"step {step} loss={float(loss):.4f} subtype={diagnostics['type']:.4f} "
                  f"role={diagnostics['role']:.4f}", flush=True)

    families = load_pool("heldout", vocab, max_instances=20000)

    def representation(items):
        with torch.no_grad():
            return model.encoder(items)

    scores = drq.heldout_scores(head, representation, embeddings, device, families, vocab)
    torch.save(head.state_dict(), HERE / "head_pretrained_directed.pt")
    (HERE / "head_pretrained_directed.json").write_text(json.dumps({
        "steps": STEPS, "lr": LR, "per_family": PER_FAMILY,
        "baseline_checkpoint": str(baseline.relative_to(REPO)),
        "heldout": scores, "train_subtype_ce": diagnostics["type"],
        "train_role_ce": diagnostics["role"],
        "n_subtypes": vocab["n_subtypes"], "n_orbits": vocab["n_orbits"],
    }, indent=1))
    print("heldout subtype macro-F1 %.4f orbit macro-F1 %.4f"
          % (scores["subtype"]["macro_f1"], scores["role"]["macro_over_families"]))


if __name__ == "__main__":
    main()
