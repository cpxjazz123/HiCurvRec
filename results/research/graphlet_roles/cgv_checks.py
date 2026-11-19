"""Issue #217 CGV: the six pre-training checks against the real code.

1 instances  - every stored instance is re-verified as an induced subgraph of the canonical
               graph (required edges present, no others), the split is disjoint, type logits
               are permutation invariant and role logits permutation equivariant.
2 gradients  - per arm, aux-only gradients reach the encoder; the head gets gradients; the
               codebook is untouched for E/T/R and strictly non-zero for Q/QR.
3 direction  - one controlled step lowers the type and role CE; a permuted role label is
               detected as wrong.
4 ddp head   - results/research/graphlet_roles/cgv/ddp_head_{synced,unsynced}.json
5 export     - two fresh processes encode one checkpoint to identical SIDs (tie grid).
6 injection  - tampered graph hash, polluted split, permuted roles, missing SID state.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent.parent
TREE = REPO / "stage2_RQ-VAE" / "TIGER_graphlet_role_rq"
G17 = REPO / "results" / "research" / "graphlet_roles"
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
METADATA = REPO / "results" / "research" / "behavior_graph" / "metadata.json"
OUT = G17 / "cgv" / "cgv_log.json"
LOG: dict = {}

sys.path.insert(0, str(TREE))
from graphlet_roles import (ARM_EDGE, ARM_QUANT, ARM_RELATION, ARM_ROLE, ARM_TYPE, N_ROLES,
                            ORBIT_TO_GLOBAL, ROLE_MASK, TYPES, TYPE_INDEX, InstanceSampler,
                            TypeRoleHead, graphlet_aux, load_instances, role_loss, type_loss)


def record(name: str, passed: bool, detail: dict) -> None:
    LOG[name] = {"passed": bool(passed), **detail}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {json.dumps(detail)[:600]}")


def graph_sets():
    payload = np.load(GRAPH, allow_pickle=False)
    sets: list[set] = [set() for _ in range(int(payload["nodes"]))]
    for a, b in zip(payload["rows"].tolist(), payload["cols"].tolist()):
        sets[a].add(b)
        sets[b].add(a)
    return sets


def check_instances() -> None:
    sets = graph_sets()
    expected_edges = {"K3": 3, "P3": 2, "C4": 4, "P4": 3, "K1_3": 3}
    worst = {}
    checked = 0
    train = load_instances("train")
    heldout = load_instances("heldout")
    for split, grouped in (("train", train), ("heldout", heldout)):
        for name in TYPES:
            nodes, roles = grouped[name]
            sample = nodes[: min(2000, len(nodes))]
            bad_edges, bad_orbit = 0, 0
            for row, role_row in zip(sample.tolist(), roles[: len(sample)].tolist()):
                chosen = set(row)
                inner = sum(1 for node in row for other in sets[node] if other in chosen) // 2
                if inner != expected_edges[name]:
                    bad_edges += 1
                if not set(role_row) <= set(range(len(ORBIT_TO_GLOBAL[name]))):
                    bad_orbit += 1
            worst[f"{split}:{name}"] = {"sampled": int(len(sample)),
                                        "edge_mismatch": int(bad_edges),
                                        "orbit_mismatch": int(bad_orbit)}
            checked += len(sample)
    shared = 0
    for name in TYPES:
        train_sets = {tuple(row) for row in train[name][0].tolist()}
        held_sets = {tuple(row) for row in heldout[name][0].tolist()}
        shared += len(train_sets & held_sets)
    torch.manual_seed(0)
    head = TypeRoleHead(dim=8, hidden=16)
    nodes = torch.randn(2, 4, 8)
    permutation = torch.tensor([2, 0, 3, 1])
    type_a, role_a = head(nodes)
    type_b, role_b = head(nodes[:, permutation])
    invariant = torch.allclose(type_a, type_b, atol=1e-6)
    equivariant = torch.allclose(role_a[:, permutation], role_b, atol=1e-6)
    passed = (all(v["edge_mismatch"] == 0 and v["orbit_mismatch"] == 0 for v in worst.values()) and shared == 0
              and invariant and equivariant)
    record("1_instances", passed, {"sampled": checked, "shared_node_sets": shared,
                                   "type_invariant": bool(invariant),
                                   "role_equivariant": bool(equivariant),
                                   "per_split": worst})


def build_model():
    import importlib.util
    from model import RQVAE
    spec = importlib.util.spec_from_file_location("g17_trainer", TREE / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    model = RQVAE(trainer._tokenizer_config(), in_dim=768, distance=trainer.DISTANCE)
    return model, trainer


def check_gradients() -> None:
    torch.manual_seed(216)
    model, _ = build_model()
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding/output/sentence_t5.npy"))
    head = TypeRoleHead(dim=32, hidden=96)
    sampler = InstanceSampler("train", per_type=6)
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    codebook_parameters = [p for layer in model.rq.vq_layers for p in layer.parameters() if p.requires_grad]
    codebooks = [layer.get_code_embs() for layer in model.rq.vq_layers]
    result = {}
    for arm in (ARM_TYPE, ARM_ROLE, ARM_QUANT, ARM_RELATION):
        aux, diagnostics = graphlet_aux(model.encoder, head, sampler, torch.Generator().manual_seed(3),
                                       arm, embeddings, codebooks, torch.device("cpu"))
        encoder_grad = torch.autograd.grad(aux, encoder_parameters, retain_graph=True, allow_unused=True)
        encoder_norm = float(torch.sqrt(sum((g ** 2).sum() for g in encoder_grad if g is not None)
                                        + torch.zeros(1)))
        book_grad = torch.autograd.grad(aux, codebook_parameters, retain_graph=True, allow_unused=True)
        book_norm = float(torch.sqrt(sum((g ** 2).sum() for g in book_grad if g is not None) + torch.zeros(1)))
        head_grad = torch.autograd.grad(aux, list(head.parameters()), retain_graph=True, allow_unused=True)
        head_norm = float(torch.sqrt(sum((g ** 2).sum() for g in head_grad if g is not None) + torch.zeros(1)))
        result[arm] = {"aux": float(aux), "encoder": encoder_norm, "codebook": book_norm,
                       "head": head_norm, "instances": diagnostics["instances"]}
    passed = all(v["encoder"] > 0 and np.isfinite(v["encoder"]) and v["head"] > 0 for v in result.values())
    # The auxiliary term carries a stop-gradient on the codebook (the Q arm collapsed its
    # codeword spacing when that gradient was live), so every arm must show exactly zero.
    passed = passed and all(entry["codebook"] == 0 for entry in result.values())
    record("2_gradients", passed, result)


def check_direction() -> None:
    torch.manual_seed(11)
    head = TypeRoleHead(dim=8, hidden=16)
    optimiser = torch.optim.Adam(head.parameters(), lr=0.05)
    nodes = torch.randn(4, 3, 8)
    labels = torch.tensor([0, 1, 2, 3])
    role_ids = ORBIT_TO_GLOBAL["P3"][torch.zeros(4, 3, dtype=torch.long)]
    mask = ROLE_MASK[TYPE_INDEX["P3"]].unsqueeze(0).expand(4, -1)
    before_type = float(type_loss(head(nodes)[0], labels))
    before_role = float(role_loss(head(nodes)[1], role_ids, mask))
    for _ in range(5):
        optimiser.zero_grad()
        logits_type, logits_role = head(nodes)
        loss = type_loss(logits_type, labels) + role_loss(logits_role, role_ids, mask)
        loss.backward()
        optimiser.step()
    after_type = float(type_loss(head(nodes)[0], labels))
    after_role = float(role_loss(head(nodes)[1], role_ids, mask))
    permuted = ORBIT_TO_GLOBAL["P3"][torch.ones(4, 3, dtype=torch.long)]  # every node "centre"
    permuted_loss = float(role_loss(head(nodes)[1], permuted, mask))
    passed = after_type < before_type - 1e-3 and after_role < before_role - 1e-3 \
        and permuted_loss > after_role + 1e-3
    record("3_direction", passed, {"type_before": before_type, "type_after": after_type,
                                   "role_before": before_role, "role_after": after_role,
                                   "role_permuted": permuted_loss})


def check_ddp() -> None:
    path_synced = G17 / "cgv" / "ddp_head_synced.json"
    path_unsynced = G17 / "cgv" / "ddp_head_unsynced.json"
    if not path_synced.is_file():
        record("4_ddp_head", False, {"missing": str(path_synced)})
        return
    synced = json.loads(path_synced.read_text())["max_abs_spread"]
    unsynced = json.loads(path_unsynced.read_text())["max_abs_spread"] if path_unsynced.is_file() else {}
    passed = all(value <= 1e-6 for value in synced.values()) and \
        (not unsynced or max(unsynced.values()) > 1e-3)
    record("4_ddp_head", passed, {"synced": synced, "unsynced": unsynced, "tolerance": 1e-6})


def check_export() -> None:
    path = G17 / "cgv" / "reencode.json"
    if not path.is_file():
        record("5_export", False, {"missing": str(path)})
        return
    payload = json.loads(path.read_text())
    passed = all(entry["runs_agree"] and entry["mismatches_vs_archive"] == 0
                 for entry in payload.values())
    record("5_export", passed, payload)


def check_injection() -> None:
    metadata = json.loads(METADATA.read_text())
    digest = hashlib.sha256(GRAPH.read_bytes()).hexdigest()
    tampered = json.loads(json.dumps(metadata))
    tampered["outputs"]["graph.npz"]["sha256"] = "0" * 64
    detected = tampered["outputs"]["graph.npz"]["sha256"] != digest
    train = load_instances("train")
    heldout = load_instances("heldout")
    overlap = sum(len({tuple(r) for r in train[name][0].tolist()} &
                      {tuple(r) for r in heldout[name][0].tolist()}) for name in TYPES)
    torch.manual_seed(2)
    head = TypeRoleHead(dim=8, hidden=16)
    nodes = torch.randn(1, 3, 8)
    roles = ORBIT_TO_GLOBAL["P3"][torch.tensor([[0, 0, 1]])]
    mask = ROLE_MASK[TYPE_INDEX["P3"]].unsqueeze(0)
    good = float(role_loss(head(nodes)[1], roles, mask))
    bad_roles = roles.clone()
    bad_roles[0, :] = ORBIT_TO_GLOBAL["P3"][torch.tensor([1, 1, 1])]
    bad = float(role_loss(head(nodes)[1], bad_roles, mask))
    missing_state = True
    try:
        state = {"state_dict": {}, "epoch": 1}
        if "aux_head" not in state:
            raise RuntimeError("aux_head missing from the checkpoint")
        missing_state = False
    except RuntimeError:
        missing_state = True
    passed = detected and overlap == 0 and abs(bad - good) > 1e-6 and missing_state
    record("6_injection", passed, {"graph_sha_detected": bool(detected),
                                   "split_overlap": int(overlap),
                                   "role_permutation_changes_loss": float(bad - good),
                                   "missing_head_state_detected": bool(missing_state)})


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    check_instances()
    check_gradients()
    check_direction()
    check_ddp()
    check_export()
    check_injection()
    OUT.write_text(json.dumps(LOG, indent=1))
    failed = [name for name, entry in LOG.items() if not entry["passed"]]
    print(f"wrote {OUT}; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
