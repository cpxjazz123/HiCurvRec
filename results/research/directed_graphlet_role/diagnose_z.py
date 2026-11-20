"""Issue #221: can a balanced, well-supported directed task be learned from frozen z?

The 46-way subtype task is imbalanced and many classes are rare, so its macro-F1
near chance cannot separate "z has no direction" from "the task is too hard".
This diagnostic keeps the label fix and shrinks the problem: the four
best-supported subtypes of one family, balanced 1:1, with a fresh head trained on
the FROZEN TIGER encoder's z.  A shuffled-label control must stay at chance.

Verdicts:
  train high, heldout high   -> z carries the direction; the 46-way failure is
                                imbalance/rarity, not the representation
  train high, heldout low    -> z overfits instances without generalising direction
  both low                   -> the frozen z does not express the direction at all
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

import drq  # noqa: E402
from directed_role import (DirectedTypeRoleHead, load_pool, load_vocabulary)  # noqa: E402
from model import RQVAE  # noqa: E402
from train_config import load_train_config  # noqa: E402

REPO = HERE.parents[2]
FAMILY = "P3"
K_SUBTYPES = 4
PER_CLASS = 1500
STEPS = 600
LR = 1e-3


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
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
    encoder = RQVAE(model_config, in_dim=embeddings.shape[1]).to(device)
    encoder.load_state_dict(torch.load(REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" /
                                       "rqvae_best.pth", map_location="cpu")["state_dict"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    vocab = load_vocabulary("directed")
    train_families = load_pool("train", vocab, max_instances=60000)
    heldout_families = load_pool("heldout", vocab, max_instances=20000)
    train_nodes, _train_orbits, train_sub = train_families[FAMILY]
    held_nodes, _held_orbits, held_sub = heldout_families[FAMILY]

    counts = np.bincount(train_sub)
    top = np.argsort(counts)[::-1][:K_SUBTYPES]
    mapping = {int(old): new for new, old in enumerate(top)}
    rng = np.random.default_rng(221)

    def sample(nodes, subtypes, per_class):
        picks = []
        for label in top:
            index = np.flatnonzero(subtypes == label)
            take = rng.choice(index, min(per_class, len(index)), replace=False)
            picks.append(take)
        chosen = np.concatenate(picks)
        rng.shuffle(chosen)
        return nodes[chosen], np.array([mapping[int(s)] for s in subtypes[chosen]])

    train_x, train_y = sample(train_nodes, train_sub, PER_CLASS)
    held_x, held_y = sample(held_nodes, held_sub, PER_CLASS // 3)
    print(f"{FAMILY}: kept subtypes {[int(v) for v in top]} with support "
          f"{[int(counts[v]) for v in top]}; train {len(train_x)} heldout {len(held_x)}")

    def features(nodes: np.ndarray) -> torch.Tensor:
        index = torch.from_numpy(nodes.reshape(-1)).long()
        with torch.no_grad():
            latent = encoder.encoder(drq.gather(embeddings, index).to(device))
        return latent.view(len(nodes), nodes.shape[1], -1)

    train_f = features(train_x)
    held_f = features(held_x)
    train_t = torch.from_numpy(train_y).long().to(device)
    held_t = torch.from_numpy(held_y).long().to(device)

    def run(labels: torch.Tensor, name: str) -> dict:
        torch.manual_seed(0)
        head = DirectedTypeRoleHead(dim=config.CODEBOOK_DIM, n_subtypes=K_SUBTYPES,
                                    n_orbits=4).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
        for step in range(1, STEPS + 1):
            logits, _ = head(train_f)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_pred = head(train_f)[0].argmax(-1)
            held_pred = head(held_f)[0].argmax(-1)

        def f1(pred, target):
            scores = []
            for label in range(K_SUBTYPES):
                tp = int(((pred == label) & (target == label)).sum())
                fp = int(((pred == label) & (target != label)).sum())
                fn = int(((pred != label) & (target == label)).sum())
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                scores.append(0.0 if precision + recall == 0 else
                              2 * precision * recall / (precision + recall))
            return float(np.mean(scores))

        record = {"control": name, "final_loss": float(loss.detach()),
                  "train_f1": f1(train_pred.cpu().numpy(), labels.cpu().numpy()),
                  "heldout_f1": f1(held_pred.cpu().numpy(), held_t.cpu().numpy()),
                  "chance": 1.0 / K_SUBTYPES}
        print(record)
        return record

    real = run(train_t, "real_labels")
    permuted = torch.from_numpy(rng.permutation(train_y)).long().to(device)
    shuffled = run(permuted, "shuffled_labels")
    results = {"family": FAMILY, "steps": STEPS,
               "top_k": {"k": int(K_SUBTYPES), "subtypes": [int(v) for v in top],
                         "support": [int(counts[v]) for v in top],
                         "real": real, "shuffled": shuffled}}

    # the two best-supported subtypes only: no rarity excuse left
    best_two = np.argsort(counts)[::-1][:2]
    pair_map = {int(old): new for new, old in enumerate(best_two)}
    keep = np.isin(train_sub, best_two)
    pair_train_x, pair_train_y = train_nodes[keep], np.array(
        [pair_map[int(s)] for s in train_sub[keep]])
    keep_h = np.isin(held_sub, best_two)
    pair_held_x, pair_held_y = held_nodes[keep_h], np.array(
        [pair_map[int(s)] for s in held_sub[keep_h]])
    rng2 = np.random.default_rng(7)
    take = {label: min(4000, int((pair_train_y == label).sum())) for label in (0, 1)}
    picks = np.concatenate([rng2.choice(np.flatnonzero(pair_train_y == label),
                                        take[label], replace=False) for label in (0, 1)])
    rng2.shuffle(picks)
    pair_f = features(pair_train_x[picks])
    pair_t = torch.from_numpy(pair_train_y[picks]).long().to(device)
    held_pick = np.concatenate([rng2.choice(np.flatnonzero(pair_held_y == label),
                                            min(1000, int((pair_held_y == label).sum())),
                                            replace=False) for label in (0, 1)])
    held_pair_f = features(pair_held_x[held_pick])
    held_pair_t = torch.from_numpy(pair_held_y[held_pick]).long().to(device)

    torch.manual_seed(0)
    head = DirectedTypeRoleHead(dim=config.CODEBOOK_DIM, n_subtypes=2, n_orbits=4).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    for step in range(1, STEPS + 1):
        logits, _ = head(pair_f)
        loss = torch.nn.functional.cross_entropy(logits, pair_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pair_record = {
            "subtypes": [int(v) for v in best_two],
            "support": [int(counts[v]) for v in best_two],
            "loss": float(loss.detach()),
            "train_accuracy": float((head(pair_f)[0].argmax(-1) == pair_t).float().mean()),
            "heldout_accuracy": float((head(held_pair_f)[0].argmax(-1) == held_pair_t).float().mean()),
            "chance": 0.5,
        }
    print(pair_record)
    results["top_two"] = pair_record
    (HERE / "diagnose_z.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
