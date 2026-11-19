"""Issue #217 evaluation: does the type/role supervision reach the discrete SID?

For every arm the heldout induced instances are scored on the exported SID:

* ``prefix_within_type`` -- the share of instances whose nodes all share an L1 (or L1+L2)
  prefix, per graphlet type;
* ``role_gap`` -- within an instance, P(share prefix | same orbit) minus
  P(share prefix | different orbits), with a cluster bootstrap over instances;
* the auxiliary head's heldout type/role macro-F1, read back from the checkpoint.

Absolute shares and the same-vs-different contrast are both reported: an arm that spreads
every item over its own codeword can show a zero gap without having learned anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
INSTANCES = REPO / "results" / "research" / "graphlet_roles" / "instances.npz"
OUT = REPO / "results" / "research" / "graphlet_roles" / "evaluation.json"
TYPES = ("K3", "P3", "C4", "P4", "K1_3")
ARMS = ("E", "T", "R", "Q", "QR")
BOOTSTRAP = 1000


def load_sid(arm: str) -> np.ndarray:
    directory = (REPO / "results" / "stage2_RQ-VAE" / f"TIGER_graphlet_role_{arm}"
                 if arm != "baseline" else REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE")
    payload = json.loads((directory / "item_sids.json").read_text())
    return np.array([payload[str(i)] for i in range(len(payload))], dtype=np.int64)


def prefix_share(codes: np.ndarray, nodes: np.ndarray, depth: int) -> np.ndarray:
    """Per instance: fraction of node pairs that share the prefix (0..1)."""
    sliced = codes[nodes, :depth]
    same = (sliced[:, :, None, :] == sliced[:, None, :, :]).all(-1)
    pairs = same.shape[1]
    return (same.sum(axis=(1, 2)) - pairs) / (pairs * (pairs - 1))


def role_gap(codes: np.ndarray, nodes: np.ndarray, roles: np.ndarray, depth: int):
    sliced = codes[nodes, :depth]
    same_code = (sliced[:, :, None, :] == sliced[:, None, :, :]).all(-1)
    same_role = roles[:, :, None] == roles[:, None, :]
    rows = np.arange(len(nodes))
    off_diagonal = ~np.eye(nodes.shape[1], dtype=bool)[None, :, :]
    same_mask = same_role & off_diagonal
    diff_mask = (~same_role) & off_diagonal
    same_share = (same_code & same_mask).sum(1) / np.maximum(same_mask.sum(1), 1)
    diff_share = (same_code & diff_mask).sum(1) / np.maximum(diff_mask.sum(1), 1)
    usable = (same_mask.sum(1) > 0) & (diff_mask.sum(1) > 0)
    return same_share, diff_share, usable


def cluster_bootstrap(values: np.ndarray, rounds: int = BOOTSTRAP, seed: int = 42) -> tuple:
    if len(values) == 0:
        return (0.0, 0.0)
    generator = np.random.default_rng(seed)
    means = [values[generator.integers(0, len(values), len(values))].mean() for _ in range(rounds)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def head_metrics(payload: dict, arm: str) -> dict:
    """Heldout type/role accuracy of the checkpoint's own head, recomputed from it."""
    import importlib.util
    import torch
    state = payload.get("aux_head")
    if not state:
        return {"available": False}
    tree = REPO / "stage2_RQ-VAE" / "TIGER_graphlet_role_rq"
    sys.path.insert(0, str(tree))
    from model import RQVAE
    from graphlet_roles import (ARM_QUANT, ARM_RELATION, ROLE_MASK, TYPE_INDEX, TYPES,
                                TypeRoleHead, load_instances, quantized_ste)
    spec = importlib.util.spec_from_file_location("g17_trainer", tree / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    model = RQVAE(trainer._tokenizer_config(), in_dim=768, distance=trainer.DISTANCE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    head = TypeRoleHead(dim=32, interaction=True)
    head.load_state_dict(state)
    head.eval()
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding/output/sentence_t5.npy"))
    codebooks = [layer.get_code_embs().detach() for layer in model.rq.vq_layers]
    heldout = load_instances("heldout")
    metrics = {"available": True, "type_accuracy": {}, "role_accuracy": {}, "instances": {}}
    with torch.no_grad():
        for name in TYPES:
            nodes, orbits = heldout[name]
            if len(nodes) == 0:
                continue
            index = torch.as_tensor(nodes.reshape(-1), dtype=torch.long)
            latent = model.encoder(embeddings[index]).reshape(len(nodes), nodes.shape[1], -1)
            features = latent
            if arm in (ARM_QUANT, ARM_RELATION):
                # Exactly the arm's own input: the cumulative soft quantisation.
                features = quantized_ste(latent, codebooks)
            type_logits, role_logits = head(features)
            metrics["type_accuracy"][name] = float(
                (type_logits.argmax(-1) == TYPE_INDEX[name]).float().mean())
            mask = ROLE_MASK[TYPE_INDEX[name]].unsqueeze(0).expand(len(nodes), -1)
            predicted = role_logits.masked_fill(~mask.unsqueeze(1), float("-inf")).argmax(-1)
            from graphlet_roles import ORBIT_TO_GLOBAL
            truth = ORBIT_TO_GLOBAL[name][torch.as_tensor(orbits.astype(np.int64))]
            metrics["role_accuracy"][name] = float((predicted == truth).float().mean())
            metrics["instances"][name] = int(len(nodes))
    return metrics


def main() -> None:
    payload = np.load(INSTANCES)
    results = {}
    for arm in ARMS + ("baseline",):
        try:
            codes = load_sid(arm)
        except FileNotFoundError:
            print(f"{arm}: no SID yet")
            continue
        checkpoint_payload = None
        entry = {"type_prefix": {}, "role": {}}
        checkpoint = (REPO / "results" / "stage2_RQ-VAE" / f"TIGER_graphlet_role_{arm}" / "rqvae_best.pth")
        if checkpoint.is_file():
            import torch
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for name in TYPES:
            keep = payload[f"{name}_split"] == 1
            nodes = payload[f"{name}_nodes"][keep]
            roles = payload[f"{name}_roles"][keep]
            if len(nodes) == 0:
                continue
            l1 = prefix_share(codes, nodes, 1)
            l2 = prefix_share(codes, nodes, 2)
            same1, diff1, usable = role_gap(codes, nodes, roles, 1)
            same2, diff2, _ = role_gap(codes, nodes, roles, 2)
            gap = (same1 - diff1)[usable]
            gap2 = (same2 - diff2)[usable]
            low, high = cluster_bootstrap(gap)
            entry["type_prefix"][name] = {
                "instances": int(len(nodes)),
                "L1_all_share": float(l1.mean()),
                "L1L2_all_share": float(l2.mean()),
            }
            entry["role"][name] = {
                "same_role_share_L1": float(same1[usable].mean()) if usable.any() else 0.0,
                "different_role_share_L1": float(diff1[usable].mean()) if usable.any() else 0.0,
                "gap_L1": float(gap.mean()) if len(gap) else 0.0,
                "gap_L1_ci": [low, high],
                "gap_L1L2": float(gap2.mean()) if len(gap2) else 0.0,
                "usable_instances": int(usable.sum()),
            }
        codes_list = codes.tolist()
        entry["codes"] = {"L1": len({row[0] for row in codes_list}),
                          "L2_prefixes": len({tuple(row[:2]) for row in codes_list}),
                          "collision": 1 - len({tuple(row[:3]) for row in codes_list}) / len(codes_list)}
        if checkpoint_payload is not None:
            entry["head"] = head_metrics(checkpoint_payload, arm)
        results[arm] = entry
    OUT.write_text(json.dumps(results, indent=1))
    header = f"{'arm':9s}{'K3 L1':>8s}{'P3 L1':>8s}{'C4 L1':>8s}{'P4 L1':>8s}{'K13 L1':>8s}{'role gap':>10s}{'coll':>8s}"
    print(header)
    for arm, entry in results.items():
        values = [entry["type_prefix"].get(name, {}).get("L1_all_share", float("nan")) for name in TYPES]
        gaps = [entry["role"].get(name, {}).get("gap_L1", 0.0) for name in ("P3", "P4", "K1_3")]
        print(f"{arm:9s}" + "".join(f"{value:8.4f}" for value in values)
              + f"{np.mean(gaps):10.4f}{entry['codes']['collision']:8.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
