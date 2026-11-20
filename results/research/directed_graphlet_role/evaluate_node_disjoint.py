"""Node-disjoint readability of the frozen vs the D1 encoder (#221).

The arm-internal `heldout_scores` splits by *instance*: an item can sit in a
train instance and in a held-out instance, so a head can memorise item->orbit
and look good without carrying any transferable structure.  This script splits
by *item* instead: a head trains only on instances whose every node is a
training item and is scored on instances made only of held-out items.
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
from directed_role import (DirectedTypeRoleHead, FAMILIES, instance_targets,  # noqa: E402
                           load_pool, load_vocabulary)
from diagnose_alignment import PooledHead, SlotHead  # noqa: E402
from model import RQVAE  # noqa: E402
from train_config import load_train_config  # noqa: E402

REPO = HERE.parents[2]
STEPS = 1500
HELD_FRACTION = 0.3
ENCODERS = {
    "baseline": REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "rqvae_best.pth",
    "D1": REPO / "results" / "stage2_RQ-VAE" / "TIGER_drq_D1" / "rqvae_best.pth",
}


def build_encoder(config, embeddings, checkpoint, device):
    model_config = type("C", (), dict(hidden_sizes=tuple(config.HIDDEN_SIZES),
                                      codebook_num=config.CODEBOOK_NUM,
                                      codebook_size=tuple(config.CODEBOOK_SIZE),
                                      codebook_dim=config.CODEBOOK_DIM, dropout=0.0,
                                      beta=config.BETA, vq_type=config.VQ_TYPE,
                                      ema_decay=config.EMA_DECAY, fix_code_embs=False,
                                      sk_epsilon=config.SK_EPSILON, sk_iters=config.SK_ITERS))()
    torch.manual_seed(42)
    encoder = RQVAE(model_config, in_dim=embeddings.shape[1]).to(device)
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_train_config(TEMPLATE / "config" / "train_config.json")
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                          "sentence_t5.npy").astype("float32"))
    vocab = load_vocabulary("directed")
    train_families = load_pool("train", vocab, max_instances=60000)
    held_families = load_pool("heldout", vocab, max_instances=20000)
    rng = np.random.default_rng(221)
    all_items = np.unique(np.concatenate([nodes.reshape(-1) for nodes, _o, _s
                                          in train_families.values()]))
    held_items = rng.permutation(all_items)[: int(HELD_FRACTION * len(all_items))]
    held_set = np.zeros(int(all_items.max()) + 1, dtype=bool)
    held_set[held_items] = True

    def partition(families):
        train_sets, test_sets = {}, {}
        for family, (nodes, orbits, subtypes) in families.items():
            inside = held_set[nodes]
            train_sets[family] = (nodes[~inside.any(1)], orbits[~inside.any(1)],
                                  subtypes[~inside.any(1)])
            test_sets[family] = (nodes[inside.all(1)], orbits[inside.all(1)],
                                 subtypes[inside.all(1)])
        return train_sets, test_sets

    train_sets, test_sets = partition(train_families)
    held_train, held_test = partition(held_families)
    results = {"held_items": int(len(held_items)), "steps": STEPS,
               "families": {}}
    for family in FAMILIES:
        results["families"][family] = {
            "train_instances": int(len(train_sets[family][0])),
            "test_instances": int(len(test_sets[family][0])),
        }
    print(json.dumps(results["families"], indent=1), flush=True)

    # balanced two-class P3, the task the diagnostics used
    family = "P3"
    nodes, orbits, subtypes = train_sets[family]
    counts = np.bincount(subtypes)
    best_two = np.argsort(counts)[::-1][:2]
    mapping = {int(old): new for new, old in enumerate(best_two)}

    def balanced(nodes, subtypes, cap):
        picks = []
        for label in best_two:
            index = np.flatnonzero(subtypes == label)
            picks.append(rng.choice(index, min(cap, len(index)), replace=False))
        chosen = np.concatenate(picks)
        rng.shuffle(chosen)
        return nodes[chosen], np.array([mapping[int(s)] for s in subtypes[chosen]])

    train_nodes, train_y = balanced(nodes, subtypes, 6000)
    test_nodes, test_y = balanced(test_sets[family][0], test_sets[family][2], 3000)

    panel = {}
    for name, checkpoint in ENCODERS.items():
        encoder = build_encoder(config, embeddings, checkpoint, device)

        def encode(batch):
            with torch.no_grad():
                index = torch.from_numpy(batch.reshape(-1)).long()
                latent = encoder.encoder(drq.gather(embeddings, index).to(device))
            return latent.view(len(batch), batch.shape[1], -1).cpu().numpy()

        x_train, x_test = encode(train_nodes), encode(test_nodes)
        y_train = torch.from_numpy(train_y).long()
        y_test = torch.from_numpy(test_y).long()
        panel[name] = {}
        for kind in ("pooled", "slot"):
            torch.manual_seed(11)
            head = (PooledHead if kind == "pooled" else SlotHead)(x_train.shape[-1])
            optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3)
            xt = torch.from_numpy(x_train)
            for _ in range(STEPS):
                loss = F.cross_entropy(head(xt), y_train)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                train_acc = float((head(xt).argmax(-1) == y_train).float().mean())
                test_acc = float((head(torch.from_numpy(x_test)).argmax(-1) == y_test).float().mean())
            panel[name][kind] = {"train_accuracy": train_acc, "test_accuracy": test_acc,
                                 "chance": 0.5}
            print(f"{name} {kind}: train {train_acc:.4f} test {test_acc:.4f}", flush=True)
    results["panel"] = panel
    results["subtypes"] = [int(v) for v in best_two]
    (HERE / "evaluate_node_disjoint.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
