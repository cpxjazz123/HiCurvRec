"""Issue #221: keep labels and head fixed, make only the ENCODER trainable.

The frozen-z diagnostic showed the 2-way balanced directed task sits at chance.
This one asks the next question: if the encoder is allowed to move (same labels,
same head, same data), does the representation become able to carry the direction,
and does a shuffled-label control stay at chance?  If the real run beats the
shuffled run, the signal is in the graph and the encoder can host it; if both stay
at chance the problem is elsewhere (head capacity or label noise), not quantisation.
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
STEPS = 800
LR = 1e-3
TRAIN_PER_CLASS = 4000
HELD_PER_CLASS = 1000


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
    baseline_state = torch.load(REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" /
                                "rqvae_best.pth", map_location="cpu")["state_dict"]

    vocab = load_vocabulary("directed")
    train_families = load_pool("train", vocab, max_instances=60000)
    held_families = load_pool("heldout", vocab, max_instances=20000)
    train_nodes, _o, train_sub = train_families[FAMILY]
    held_nodes, _oh, held_sub = held_families[FAMILY]
    counts = np.bincount(train_sub)
    best_two = np.argsort(counts)[::-1][:2]
    mapping = {int(old): new for new, old in enumerate(best_two)}
    print(f"{FAMILY}: subtypes {best_two.tolist()} support {counts[best_two].tolist()}")

    rng = np.random.default_rng(7)

    def balanced(nodes, subtypes):
        picks = []
        for label in best_two:
            index = np.flatnonzero(subtypes == label)
            picks.append(rng.choice(index, min(TRAIN_PER_CLASS if nodes is train_nodes else HELD_PER_CLASS,
                                               len(index)), replace=False))
        chosen = np.concatenate(picks)
        rng.shuffle(chosen)
        return nodes[chosen], np.array([mapping[int(s)] for s in subtypes[chosen]])

    train_x, train_y = balanced(train_nodes, train_sub)
    held_x, held_y = balanced(held_nodes, held_sub)

    def build_encoder(trainable: bool) -> RQVAE:
        torch.manual_seed(42)
        encoder = RQVAE(model_config, in_dim=embeddings.shape[1]).to(device)
        encoder.load_state_dict(baseline_state)
        for parameter in encoder.parameters():
            parameter.requires_grad_(trainable)
        return encoder

    def features(encoder, nodes: np.ndarray) -> torch.Tensor:
        index = torch.from_numpy(nodes.reshape(-1)).long()
        latent = encoder.encoder(drq.gather(embeddings, index).to(device))
        return latent.view(len(nodes), nodes.shape[1], -1)

    def run(trainable: bool, labels: np.ndarray, name: str) -> dict:
        encoder = build_encoder(trainable)
        torch.manual_seed(0)
        head = DirectedTypeRoleHead(dim=config.CODEBOOK_DIM, n_subtypes=2, n_orbits=4).to(device)
        parameters = list(head.parameters())
        if trainable:
            parameters += [p for p in encoder.encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=LR)
        target = torch.from_numpy(labels).long().to(device)
        for step in range(1, STEPS + 1):
            logits, _ = head(features(encoder, train_x))
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 200 == 0:
                print(f"  {name} step {step} loss {float(loss):.4f}", flush=True)
        with torch.no_grad():
            encoder.eval()
            train_acc = float((head(features(encoder, train_x))[0].argmax(-1) == target)
                              .float().mean())
            held_target = torch.from_numpy(held_y).long().to(device)
            held_acc = float((head(features(encoder, held_x))[0].argmax(-1) == held_target)
                             .float().mean())
        record = {"condition": name, "encoder_trainable": trainable,
                  "loss": float(loss.detach()), "train_accuracy": train_acc,
                  "heldout_accuracy": held_acc, "chance": 0.5}
        print(record, flush=True)
        return record

    torch.manual_seed(123)
    shuffled = np.random.default_rng(2026).permutation(train_y)
    results = {
        "family": FAMILY,
        "subtypes": [int(v) for v in best_two],
        "support": [int(counts[v]) for v in best_two],
        "steps": STEPS,
        "frozen_real": run(False, train_y, "frozen_real"),
        "frozen_shuffled": run(False, shuffled, "frozen_shuffled"),
        "joint_real": run(True, train_y, "joint_real"),
        "joint_shuffled": run(True, shuffled, "joint_shuffled"),
    }
    (HERE / "diagnose_joint.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
