"""Does the hard SID keep the structure the continuous code carries? (#221)

Same subgraph encoder, same directed adjacency, same batches and the same
initialisation; only the node feature changes:

    semantic   the stage1 sentence embedding (what the RQ-VAE reads)
    z          the encoder output before quantisation (32-d)
    q          the codebook sum of the hard three-token code (32-d)

`z` vs `q` is the quantisation gap: everything the encoder knows minus what
survives the discrete code.  Both orbit numbers are unconditional (the head is
never told the true subtype).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template"
sys.path.insert(0, str(TEMPLATE))

import drq  # noqa: E402
from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import FAMILIES, instance_targets, load_pool, load_vocabulary  # noqa: E402
from model import RQVAE  # noqa: E402
from subgraph_predictor import (HIDDEN, Readout, SubgraphEncoder, adjacency_tensor,  # noqa: E402
                                macro_f1)
from train_config import load_train_config  # noqa: E402

REPO = HERE.parents[2]
TRAIN_CAP = 20000
TEST_CAP = 5000
STEPS = 1500
BATCH = 512
SEED = 20261120
ARMS = {
    "base": REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "rqvae_best.pth",
    "D1": REPO / "results" / "stage2_RQ-VAE" / "TIGER_drq_D1" / "rqvae_best.pth",
}


def build_rqvae(config, in_dim, checkpoint, device):
    model_config = type("C", (), dict(hidden_sizes=tuple(config.HIDDEN_SIZES),
                                      codebook_num=config.CODEBOOK_NUM,
                                      codebook_size=tuple(config.CODEBOOK_SIZE),
                                      codebook_dim=config.CODEBOOK_DIM, dropout=0.0,
                                      beta=config.BETA, vq_type=config.VQ_TYPE,
                                      ema_decay=config.EMA_DECAY, fix_code_embs=False,
                                      sk_epsilon=config.SK_EPSILON, sk_iters=config.SK_ITERS))()
    torch.manual_seed(42)
    model = RQVAE(model_config, in_dim=in_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def item_vectors(model, semantic, chunk: int = 4096):
    z_parts, q_parts = [], []
    for start in range(0, len(semantic), chunk):
        batch = semantic[start:start + chunk]
        encoded = model.encoder(batch)
        quantized, _parts, _unused, _tokens = model.rq(encoded)
        z_parts.append(encoded.cpu())
        q_parts.append(quantized.cpu())
    return torch.cat(z_parts), torch.cat(q_parts)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    config = load_train_config(TEMPLATE / "config" / "train_config.json")
    semantic = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                        "sentence_t5.npy").astype("float32")).to(device)
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    edge_keys = rows.astype(np.int64) * n_items + cols
    vocab = load_vocabulary("directed")
    train_families = load_pool("train", vocab, max_instances=TRAIN_CAP)
    test_families = load_pool("heldout", vocab, max_instances=TEST_CAP)
    prepared = {}
    for family in FAMILIES:
        tn, to, ts = train_families[family]
        sn, so, ss = test_families[family]
        prepared[family] = {
            "train": (tn, instance_targets(family, ts, to, vocab)),
            "test": (sn, instance_targets(family, ss, so, vocab)),
        }
    adjacency = {(family, split): torch.from_numpy(
        adjacency_tensor(prepared[family][split][0], edge_keys, n_items)).to(device)
        for family in FAMILIES for split in ("train", "test")}

    results = {"arms": {}, "steps": STEPS, "train_cap": TRAIN_CAP, "test_cap": TEST_CAP}
    for arm, checkpoint in ARMS.items():
        model = build_rqvae(config, semantic.shape[1], checkpoint, device)
        z, q = item_vectors(model, semantic)
        sources = {"semantic": semantic.cpu(), "z": z, "q": q}
        results["arms"][arm] = {}
        for source, table in sources.items():
            features = table.to(device)
            torch.manual_seed(7)
            encoder = SubgraphEncoder(features.shape[1]).to(device)
            readout = Readout(HIDDEN, vocab["n_subtypes"], vocab["n_orbits"]).to(device)
            optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(readout.parameters()),
                                          lr=1e-3)
            generator = torch.Generator().manual_seed(SEED)
            order = [(family, split) for split in ("train",) for family in FAMILIES]
            for step in range(1, STEPS + 1):
                family, split = order[step % len(order)]
                nodes, targets = prepared[family][split]
                kept = np.flatnonzero(targets["subtype"] >= 0)
                draw = torch.randint(0, len(kept), (min(BATCH, len(kept)),),
                                     generator=generator).numpy()
                pick = torch.from_numpy(kept[draw]).to(device)
                index = torch.from_numpy(nodes).long().to(device)[pick]
                h = encoder(features[index], adjacency[family, split][pick])
                type_logits, role_logits = readout(h)
                target_sub = torch.from_numpy(targets["subtype"]).to(device)[pick]
                target_orbit = torch.from_numpy(targets["orbit"]).to(device)[pick]
                loss = (F.cross_entropy(type_logits, target_sub.clamp(min=0))
                        + F.cross_entropy(role_logits.reshape(-1, role_logits.shape[-1]),
                                          target_orbit.reshape(-1)))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            per_family = {}
            with torch.no_grad():
                for family in FAMILIES:
                    nodes, targets = prepared[family]["test"]
                    kept = np.flatnonzero(targets["subtype"] >= 0)
                    index = torch.from_numpy(nodes[kept]).long().to(device)
                    h = encoder(features[index], adjacency[family, "test"][kept])
                    type_logits, role_logits = readout(h)
                    per_family[family] = {
                        "subtype_macro_f1": macro_f1(type_logits.argmax(-1).cpu().numpy(),
                                                     targets["subtype"][kept], vocab["n_subtypes"]),
                        "orbit_macro_f1": macro_f1(role_logits.argmax(-1).cpu().numpy().reshape(-1),
                                                   targets["orbit"][kept].reshape(-1),
                                                   vocab["n_orbits"]),
                        "instances": int(len(kept)),
                    }
            results["arms"][arm][source] = {
                "subtype_macro_f1": float(np.mean([v["subtype_macro_f1"] for v in per_family.values()])),
                "orbit_macro_f1": float(np.mean([v["orbit_macro_f1"] for v in per_family.values()])),
                "per_family": per_family,
            }
            summary = results["arms"][arm][source]
            print(f"{arm} {source}: subtype {summary['subtype_macro_f1']:.4f} "
                  f"orbit {summary['orbit_macro_f1']:.4f}", flush=True)
    (HERE / "quantisation_gap.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
