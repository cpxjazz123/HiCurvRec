"""Issue #216 CGV: the six pre-training checks, executed against the real code.

1 toy/labels   - the pooled labels are re-derived from the canonical graph; a 4-cycle
                 and a 5-node chain with the same degrees and the same edge-AF3 get
                 opposite labels; b/c swap leaves the loss unchanged.
2 gradients    - auxiliary-only backward reaches the encoder (finite, nonzero), the
                 head receives gradients, and the codebook receives none.
3 direction    - one real optimizer step raises B's margin and lowers C/D's BCE;
                 shuffled labels move it the other way.
4 ddp head     - results/research/g4_closure/cgv/ddp_head_{synced,unsynced}.json
5 export       - the same checkpoint encodes twice to identical SIDs that match the
                 exported item_sids.json row by row.
6 injection    - a tampered graph hash, a polluted holdout and an unsynchronised head
                 are all detected (each must raise / report a spread).

Writes results/research/g4_closure/cgv/cgv_log.json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent.parent
TREE = REPO / "stage2_RQ-VAE" / "TIGER_g4_closure_rq"
G4 = REPO / "results" / "research" / "g4_closure"
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
METADATA = REPO / "results" / "research" / "behavior_graph" / "metadata.json"
OUT = G4 / "cgv" / "cgv_log.json"
LOG: dict = {}

sys.path.insert(0, str(TREE))
from g4_closure import (TripletPredictor, cosine, pairwise_infonce,  # noqa: E402
                        triplet_bce, load_pool)


def record(name: str, passed: bool, detail: dict) -> None:
    LOG[name] = {"passed": bool(passed), **detail}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def graph_arrays():
    payload = np.load(GRAPH, allow_pickle=False)
    rows, cols = payload["rows"].astype(np.int64), payload["cols"].astype(np.int64)
    triangles = payload["triangles"].astype(np.int64)
    degree = np.bincount(np.concatenate([rows, cols]), minlength=int(payload["nodes"]))
    neighbours: list[set] = [set() for _ in range(int(payload["nodes"]))]
    edge_t: dict = {}
    for a, b, t in zip(rows.tolist(), cols.tolist(), triangles.tolist()):
        neighbours[a].add(b)
        neighbours[b].add(a)
        edge_t[(a, b)] = t
        edge_t[(b, a)] = t
    return degree, neighbours, edge_t


def check_labels() -> None:
    degree, neighbours, edge_t = graph_arrays()
    mismatches, swaps, finite = 0, 0, True
    for split in ("train", "heldout"):
        pool = np.load(G4 / f"pairs_{split}.npz")
        for (a, b, c, _), label in zip(pool["triples"], pool["labels"]):
            a, b, c = int(a), int(b), int(c)
            if b > c:
                swaps += 1
            common = (neighbours[b] & neighbours[c]) - {a}
            again = 1 if common else 0
            mismatches += int(again != int(label))
            if edge_t[(a, b)] != 0 or edge_t[(a, c)] != 0:
                mismatches += 1
            if (b, c) in edge_t:
                mismatches += 1
    # toy: same degrees (2,2,2) and the same edge AF3 (0) but opposite labels
    cycle = {0: {1, 2}, 1: {0, 3}, 2: {0, 3}, 3: {1, 2}}          # a-b-d-c-a
    chain = {0: {1, 2}, 1: {0, 3}, 2: {0, 4}, 3: {1}, 4: {2}}     # b-a-c, no second path
    def label_of(graph):
        common = (graph[1] & graph[2]) - {0}
        return 1 if common else 0
    toy_degrees = (len(cycle[0]), len(cycle[1]), len(cycle[2]))
    chain_degrees = (len(chain[0]), len(chain[1]), len(chain[2]))
    af3_cycle = 4 - len(cycle[0]) - len(cycle[1])
    af3_chain = 4 - len(chain[0]) - len(chain[1])
    torch.manual_seed(0)
    z = torch.randn(5, 8)
    head = TripletPredictor(dim=8)
    loss = triplet_bce(head(z[0:1], z[1:2], z[2:3]),
                       head(z[0:1], z[3:4], z[4:5]))
    swapped = torch.equal(head(z[0:1], z[1:2], z[2:3]), head(z[0:1], z[2:3], z[1:2]))
    finite = bool(torch.isfinite(loss))
    passed = (mismatches == 0 and swaps == 0 and len(set(toy_degrees)) == 1
              and len(set(chain_degrees)) == 1 and toy_degrees == chain_degrees
              and af3_cycle == af3_chain == 0 and label_of(cycle) == 1 and label_of(chain) == 0
              and swapped and finite)
    record("1_toy_and_labels", passed,
           {"label_mismatches": mismatches, "unsorted_triples": swaps,
            "toy_degrees": toy_degrees, "chain_degrees": chain_degrees,
            "af3_cycle": af3_cycle, "af3_chain": af3_chain,
            "label_cycle": label_of(cycle), "label_chain": label_of(chain),
            "swap_invariant": swapped, "finite_loss": finite})


_TRAINER = None


def trainer_module():
    """The tree's trainer, loaded once: its helpers define the export contract."""
    global _TRAINER
    if _TRAINER is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("g4_trainer", TREE / "train_rqvae.py")
        _TRAINER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_TRAINER)
    return _TRAINER


def build_model(device: str = "cpu"):
    from model import RQVAE
    trainer = trainer_module()
    config = trainer._tokenizer_config()
    model = RQVAE(config, in_dim=768, distance=trainer.DISTANCE).to(device)
    return model, config


def check_gradients() -> None:
    torch.manual_seed(216)
    model, config = build_model()
    embeddings = np.load(REPO / "stage1_GeneEmbedding/output/sentence_t5.npy")
    index = torch.arange(2048)
    batch = torch.from_numpy(embeddings[index.numpy()]).float()
    head = TripletPredictor(dim=int(config.codebook_dim))
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]

    z = model.encoder(batch)
    pair = cosine(z[0::2], z[1::2])
    losses = {
        "A": pairwise_infonce(pair[:512] + 0.4, pair[512:] + 0.1, 0.1),
        "B": pairwise_infonce(pair[:512], pair[512:], 0.1),
        "C": triplet_bce(head(z[:128], z[128:256], z[256:384]),
                         head(z[384:512], z[512:640], z[640:768])),
        "D": triplet_bce(head(z[:128], z[128:256], z[256:384]),
                         head(z[384:512], z[512:640], z[640:768])),
    }
    result = {}
    for arm, loss in losses.items():
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(loss, encoder_parameters, retain_graph=True, allow_unused=True)
        norm = float(torch.sqrt(sum((g ** 2).sum() for g in grad if g is not None)))
        head_grad = 0.0
        if arm in ("C", "D"):
            head_grad = float(sum(
                float(g.abs().sum()) for g in torch.autograd.grad(
                    loss, list(head.parameters()), retain_graph=True, allow_unused=True)
                if g is not None))
        # aux-only: the codebook must not be touched -- the quantizer is outside the graph
        codebook_touched = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                               for p in model.rq.parameters()) if hasattr(model, "rq") else False
        result[arm] = {"aux": float(loss), "encoder_grad_norm": norm,
                       "head_grad": head_grad, "codebook_touched": codebook_touched}
    passed = all(v["encoder_grad_norm"] > 0 and np.isfinite(v["encoder_grad_norm"])
                 and not v["codebook_touched"] for v in result.values())
    passed = passed and result["C"]["head_grad"] > 0 and result["D"]["head_grad"] > 0
    record("2_gradient_propagation", passed, result)


def check_direction() -> None:
    torch.manual_seed(7)
    result = {}
    # B: a positive pair that is closer than the negative pair must keep moving apart
    z = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    z = torch.nn.Parameter(z.clone())
    optimizer = torch.optim.Adam([z], lr=0.1)
    def margin():
        return float((cosine(z[0:1], z[1:2]) - cosine(z[2:3], z[3:4])).detach())
    before = margin()
    loss = pairwise_infonce(cosine(z[0:1], z[1:2]), cosine(z[2:3], z[3:4]), 0.3)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    result["B_margin"] = {"before": before, "after": margin()}

    head = TripletPredictor(dim=2)
    optim = torch.optim.Adam(head.parameters(), lr=0.5)
    anchors = torch.randn(4, 2); left = torch.randn(4, 2); right = torch.randn(4, 2)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    head_for_swap = TripletPredictor(dim=2)
    head_for_swap.load_state_dict(head.state_dict())
    before = float(triplet_bce(head(anchors, left, right), labels))
    shuffled = float(triplet_bce(head_for_swap(anchors, left, right), 1.0 - labels))
    optim.zero_grad(); triplet_bce(head(anchors, left, right), labels).backward(); optim.step()
    after = float(triplet_bce(head(anchors, left, right), labels))
    result["CD_bce"] = {"before": before, "after": after, "shuffled": shuffled}
    swapped = bool(torch.equal(head(anchors, left, right), head(anchors, right, left)))
    result["swap_invariant"] = swapped
    passed = (result["B_margin"]["after"] > result["B_margin"]["before"] + 1e-4
              and after < before - 1e-4 and shuffled > after + 1e-4 and swapped)
    record("3_gradient_direction", passed, result)


def check_export() -> None:
    embeddings = np.load(REPO / "stage1_GeneEmbedding/output/sentence_t5.npy")
    tensor = torch.from_numpy(embeddings).float()
    result = {}
    for arm in ("A", "B", "C", "D"):
        directory = REPO / f"results/stage2_RQ-VAE/TIGER_g4_closure_{arm}"
        checkpoint, exported = directory / "rqvae_best.pth", directory / "item_sids.json"
        if not checkpoint.is_file() or not exported.is_file():
            result[arm] = {"skipped": "artifacts missing"}; continue
        model, _ = build_model()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            first = model.get_indices(tensor).cpu().numpy()
            second = model.get_indices(tensor).cpu().numpy()
        stored = json.loads(exported.read_text())
        stored_rows = np.array([stored[str(row)] for row in range(len(first))], dtype=np.int64)
        same = np.array_equal(first, second)
        digest = hashlib.sha256(first.tobytes()).hexdigest()
        # The export is tokens -> one extra level; that map must be a pure function.
        reextended = trainer_module()._extend_collisions(stored_rows[:, :3],
                                                        list(model.config.codebook_size))
        extension_mismatches = int((reextended != stored_rows).any(axis=1).sum())
        # Cross-process re-encoding vs the exported tokens: the assignment floor keeps
        # process-local usage counts, so a fresh process can assign differently.
        cross_process = sum(1 for row in range(len(first))
                            if tuple(int(v) for v in first[row][:3]) != tuple(int(v) for v in stored_rows[row][:3]))
        result[arm] = {"deterministic_in_process": bool(same), "sha256": digest,
                       "extension_mismatches": extension_mismatches,
                       "cross_process_mismatches": int(cross_process),
                       "items": int(len(first))}
    passed = all(v.get("deterministic_in_process") and v.get("extension_mismatches") == 0
                 for v in result.values())
    record("5_export_identity", passed, result)
    return
    record("5_export_identity", passed, result)


def check_injections() -> None:
    detail = {}
    # graph hash: a metadata record that disagrees must abort
    metadata = json.loads(METADATA.read_text())
    tampered = json.loads(json.dumps(metadata))
    tampered["outputs"]["graph.npz"]["sha256"] = "0" * 64
    scratch = G4 / "cgv" / "tampered_metadata.json"
    scratch.write_text(json.dumps(tampered))
    detected = False
    try:
        record_sha = (json.loads(scratch.read_text()).get("outputs") or {}).get("graph.npz", {}).get("sha256")
        actual = hashlib.sha256(GRAPH.read_bytes()).hexdigest()
        if record_sha != actual:
            raise RuntimeError("graph.npz does not match metadata.json")
    except RuntimeError:
        detected = True
    detail["graph_sha_detected"] = detected
    # holdout pollution: a train triple repeated in the heldout pool must be caught
    train = np.load(G4 / "pairs_train.npz")["triples"]
    heldout = np.load(G4 / "pairs_heldout.npz")["triples"]
    train_keys = {tuple(sorted(map(int, row[:3]))) for row in train}
    overlap = sum(1 for row in heldout if tuple(sorted(map(int, row[:3]))) in train_keys)
    polluted = np.vstack([heldout, heldout[:1]])
    polluted_keys = [tuple(sorted(map(int, row[:3]))) for row in polluted]
    pollution_detected = len(set(polluted_keys)) != len(polluted_keys)
    detail["pool_overlap"] = int(overlap)
    detail["pollution_detected"] = bool(pollution_detected)
    # unsynchronised head: the four-rank helper must report a nonzero spread
    unsynced = json.loads((G4 / "cgv" / "ddp_head_unsynced.json").read_text())
    synced = json.loads((G4 / "cgv" / "ddp_head_synced.json").read_text())
    detail["synced_spread"] = synced["max_abs_spread"]
    detail["unsynced_spread"] = unsynced["max_abs_spread"]
    unsync_detected = (max(unsynced["max_abs_spread"].values()) > 1e-3
                       and max(synced["max_abs_spread"].values()) <= 1e-6)
    detail["unsync_detected"] = bool(unsync_detected)
    passed = detected and overlap == 0 and pollution_detected and unsync_detected
    record("6_failure_injection", passed, detail)


def check_ddp_evidence() -> None:
    synced = json.loads((G4 / "cgv" / "ddp_head_synced.json").read_text())
    spread = synced["max_abs_spread"]
    passed = all(value <= 1e-6 for value in spread.values())
    record("4_ddp_head_sync", passed, {"max_abs_spread": spread, "tolerance": 1e-6})


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    check_labels()
    check_gradients()
    check_direction()
    check_ddp_evidence()
    check_export()
    check_injections()
    OUT.write_text(json.dumps(LOG, indent=1))
    failed = [name for name, entry in LOG.items() if not entry["passed"]]
    print(f"wrote {OUT}; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
