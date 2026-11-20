"""Does giving the predictor directed structure information fix the task? (#221)

Same labels, same instances, same head initialisation, same optimiser: only the
per-node input changes.

  z                 frozen TIGER encoder output (the current predictor input)
  z+degree          in/out/mutual/total degree -- the "node hotness" control
  z+dgdv(directed)  directed graphlet degree vector over triads
  z+dgdv(undirected) the same counts with every direction erased
  z+dgdv(shuffled)  the same counts after shuffling edge directions
  dgdv(directed)    structure only, no semantic embedding

The held-out graph excludes every edge inside a held-out instance, so a
held-out instance's own edges are never part of its features.
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
from directed_role import load_pool, load_vocabulary  # noqa: E402
from diagnose_alignment import PooledHead, SlotHead  # noqa: E402
from model import RQVAE  # noqa: E402
from train_config import load_train_config  # noqa: E402

REPO = HERE.parents[2]
FAMILY = "P3"
STEPS = 600
SIZES = (1024, 4096)
VARIANTS = (
    ("z", ()),
    ("z+degree", ("degrees",)),
    ("z+dgdv_directed", ("directed_counts",)),
    ("z+dgdv_undirected", ("undirected_counts",)),
    ("z+dgdv_shuffled", ("shuffled_counts",)),
    ("dgdv_directed", ("directed_counts", "no_z")),
)


def load_blocks() -> dict:
    payload = np.load(HERE / "dgdv.npz")
    return {key: payload[key] for key in payload.files}


def standardise(block: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.log1p(np.asarray(block, dtype=np.float64)[index])
    return values.mean(0), values.std(0) + 1e-6


def main() -> None:
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
    state = torch.load(REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" /
                       "rqvae_best.pth", map_location="cpu")["state_dict"]
    torch.manual_seed(42)
    encoder = RQVAE(model_config, in_dim=embeddings.shape[1]).to(device)
    encoder.load_state_dict(state)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    blocks = load_blocks()
    vocab = load_vocabulary("directed")
    train_families = load_pool("train", vocab, max_instances=60000)
    held_families = load_pool("heldout", vocab, max_instances=20000)
    train_nodes, _o, train_sub = train_families[FAMILY]
    held_nodes, _oh, held_sub = held_families[FAMILY]
    counts = np.bincount(train_sub)
    best_two = np.argsort(counts)[::-1][:2]
    mapping = {int(old): new for new, old in enumerate(best_two)}
    rng = np.random.default_rng(7)

    def balanced(nodes, subtypes, total):
        picks = []
        for label in best_two:
            index = np.flatnonzero(subtypes == label)
            picks.append(rng.choice(index, min(total // 2, len(index)), replace=False))
        chosen = np.concatenate(picks)
        rng.shuffle(chosen)
        return nodes[chosen], np.array([mapping[int(s)] for s in subtypes[chosen]])

    def encode(nodes: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            index = torch.from_numpy(nodes.reshape(-1)).long()
            latent = encoder.encoder(drq.gather(embeddings, index).to(device))
        return latent.view(len(nodes), nodes.shape[1], -1).cpu().numpy()

    train_nodes_bal, train_y = balanced(train_nodes, train_sub, max(SIZES))
    held_nodes_bal, held_y = balanced(held_nodes, held_sub, 4000)
    held_all = encode(held_nodes_bal)
    held_blocks = {name: (np.log1p(np.asarray(block, dtype=np.float64)[held_nodes_bal])
                          ) for name, block in blocks.items()}

    results = {"family": FAMILY, "subtypes": [int(v) for v in best_two],
               "support": [int(counts[v]) for v in best_two],
               "steps": STEPS, "runs": []}
    for total in SIZES:
        nodes = train_nodes_bal[:total]
        labels = train_y[:total]
        train_z = encode(nodes)
        train_moments = {name: standardise(block, nodes) for name, block in blocks.items()}
        for variant, sources in VARIANTS:
            use_z = "no_z" not in sources
            parts_train, parts_held = ([train_z], [held_all]) if use_z else ([], [])
            for source in sources:
                if source == "no_z":
                    continue
                mean, std = train_moments[source]
                parts_train.append((np.log1p(np.asarray(blocks[source],
                                                         dtype=np.float64)[nodes]) - mean) / std)
                parts_held.append((held_blocks[source] - mean) / std)
            x_train = torch.from_numpy(np.concatenate(parts_train, -1).astype("float32"))
            x_held = torch.from_numpy(np.concatenate(parts_held, -1).astype("float32"))
            y_train = torch.from_numpy(labels).long()
            y_held = torch.from_numpy(held_y).long()
            for kind in ("pooled", "slot"):
                torch.manual_seed(11)
                head = (PooledHead if kind == "pooled" else SlotHead)(x_train.shape[-1])
                optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3)
                for _ in range(STEPS):
                    loss = F.cross_entropy(head(x_train), y_train)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    train_acc = float((head(x_train).argmax(-1) == y_train).float().mean())
                    held_acc = float((head(x_held).argmax(-1) == y_held).float().mean())
                row = {"n": int(total), "variant": variant, "head": kind,
                       "dim": int(x_train.shape[-1]), "train_accuracy": train_acc,
                       "heldout_accuracy": held_acc, "chance": 0.5}
                results["runs"].append(row)
                print(json.dumps(row), flush=True)
    (HERE / "diagnose_structure.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
