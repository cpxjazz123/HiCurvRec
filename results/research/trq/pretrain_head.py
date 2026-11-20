"""Issue #220 arm E: pre-train the Type+Role head on z, validate, save, freeze.

The head is trained against the FROZEN baseline encoder (TIGER_RQ-VAE's
checkpoint) on the TRAIN instances; the heldout score is recorded with the same
code that later scores every arm.  Nothing here touches the RQ-VAE being trained.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template"
sys.path.insert(0, str(TEMPLATE))

import trq  # noqa: E402
from graphlet_roles import (ORBIT_TO_GLOBAL, ROLE_MASK, TYPE_INDEX, TYPES, InstanceSampler,  # noqa: E402
                            TypeRoleHead, load_instances)
from model import RQVAE  # noqa: E402
from train_config import load_train_config  # noqa: E402

REPO = HERE.parents[2]
STEPS = 3000
BATCH_PER_TYPE = 32
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

    head = TypeRoleHead(dim=config.CODEBOOK_DIM).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    sampler = InstanceSampler("train", BATCH_PER_TYPE)
    generator = torch.Generator().manual_seed(42)
    for step in range(1, STEPS + 1):
        draw = trq.instance_draw(sampler, generator)
        features, roles, types, masks = {}, {}, {}, {}
        for name in TYPES:
            if name not in draw:
                continue
            nodes, orbits = draw[name]
            count, width = nodes.shape
            index = torch.from_numpy(nodes.reshape(-1)).long()
            with torch.no_grad():
                latent = model.encoder(trq.gather(embeddings, index).to(device))
            features[name] = latent.view(count, width, -1)
            roles[name] = ORBIT_TO_GLOBAL[name][torch.from_numpy(orbits.astype(np.int64))].to(device)
            types[name] = torch.full((count,), TYPE_INDEX[name], dtype=torch.long, device=device)
            masks[name] = ROLE_MASK[TYPE_INDEX[name]].unsqueeze(0).expand(count, -1).to(device)
        loss, diagnostics = trq.joint_loss(head, features, roles, types, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 250 == 0:
            print(f"step {step} loss={float(loss):.4f} type={diagnostics['type']:.4f} "
                  f"role={diagnostics['role']:.4f}", flush=True)

    heldout = load_instances("heldout")

    def representation(items):
        with torch.no_grad():
            return model.encoder(items)

    scores = trq.heldout_scores(head, representation, embeddings, device, heldout)
    out_dir = HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out_dir / "head_pretrained.pt")
    record = {"steps": STEPS, "lr": LR, "batch_per_type": BATCH_PER_TYPE,
              "baseline_checkpoint": str(baseline.relative_to(REPO)),
              "heldout": scores,
              "train_type_ce": diagnostics["type"], "train_role_ce": diagnostics["role"]}
    (out_dir / "head_pretrained.json").write_text(json.dumps(record, indent=1))
    print("heldout type macro-F1 %.4f role macro-over-types %.4f"
          % (scores["type"]["macro_f1"], scores["role"]["macro_over_types"]))
    for name, row in scores["role"]["per_type"].items():
        print(f"  {name}: role macro-F1 {row['macro_f1']:.4f} ({row['classes']} classes)")


if __name__ == "__main__":
    main()
