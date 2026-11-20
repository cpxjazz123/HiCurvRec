"""Issue #222 A: one fixed Teacher run, saved as an artefact with its receipts.

Same definition, data, seed and budget as `directed_graphlet_role/subgraph_predictor.py`
(2-layer in/out separated message passing, hidden 128, 1500 steps, batch 512,
seed 20261120, the directed instance pool).  The three input variants are trained
again here so the numbers beside the checkpoint come from the checkpoint's own run
rather than from an older printout; only `directed` is saved as the Teacher.

Outputs
    teacher/teacher_directed_v1.pth      payload (format, config, vocab, weights)
    teacher/teacher_metadata.json        SHA, pool/graph SHA, config, metrics
    teacher/teacher_probe.pt             fixed batch, logits and targets for reload checks
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
ROLE_DIR = REPO / "results" / "research" / "directed_graphlet_role"
sys.path.insert(0, str(ROLE_DIR))

from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import FAMILIES, instance_targets, load_pool, load_vocabulary  # noqa: E402
from teacher_model import build_teacher, save_teacher, teacher_config  # noqa: E402

TRAIN_CAP = 20000
TEST_CAP = 5000
HIDDEN = 128
LAYERS = 2
STEPS = 1500
BATCH = 512
SEED = 20261120
OUT_DIR = HERE / "teacher"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def macro_f1(prediction: np.ndarray, target: np.ndarray, n_classes: int) -> float:
    scores = []
    for label in range(n_classes):
        tp = int(((prediction == label) & (target == label)).sum())
        fp = int(((prediction == label) & (target != label)).sum())
        fn = int(((prediction != label) & (target == label)).sum())
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def adjacency_tensor(nodes: np.ndarray, edge_keys: np.ndarray, n_items: int) -> np.ndarray:
    count, width = nodes.shape
    out = np.zeros((count, width, width), dtype=np.float32)
    for i in range(width):
        for j in range(width):
            if i == j:
                continue
            keys = nodes[:, i].astype(np.int64) * n_items + nodes[:, j]
            out[:, i, j] = np.isin(keys, edge_keys).astype(np.float32)
    return out


def shuffle_directions(a: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = a.copy()
    for i in range(a.shape[1]):
        for j in range(i + 1, a.shape[2]):
            flip = rng.random(len(a)) < 0.5
            out[:, i, j] = np.where(flip, a[:, j, i], a[:, i, j])
            out[:, j, i] = np.where(flip, a[:, i, j], a[:, j, i])
    return out


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    edge_keys = rows.astype(np.int64) * n_items + cols
    semantic = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                        "sentence_t5.npy").astype("float32")).to(device)
    vocab = load_vocabulary("directed")
    train_families = load_pool("train", vocab, max_instances=TRAIN_CAP)
    test_families = load_pool("heldout", vocab, max_instances=TEST_CAP)

    prepared = {}
    for family in FAMILIES:
        train_nodes, train_orbits, train_subtypes = train_families[family]
        test_nodes, test_orbits, test_subtypes = test_families[family]
        prepared[family] = {
            "train": (train_nodes, instance_targets(family, train_subtypes, train_orbits, vocab)),
            "test": (test_nodes, instance_targets(family, test_subtypes, test_orbits, vocab)),
        }
        print(f"{family}: train {len(train_nodes)} test {len(test_nodes)}", flush=True)

    cache = {}
    for family in FAMILIES:
        for split in ("train", "test"):
            nodes = prepared[family][split][0]
            base = adjacency_tensor(nodes, edge_keys, n_items)
            cache[(family, split, "directed")] = torch.from_numpy(base).to(device)
            cache[(family, split, "no_graph")] = torch.zeros(
                (len(nodes), nodes.shape[1], nodes.shape[1]), dtype=torch.float32, device=device)
            cache[(family, split, "shuffled")] = torch.from_numpy(
                shuffle_directions(base, SEED)).to(device)

    results = {}
    directed_teacher = None
    for mode in ("directed", "no_graph", "shuffled"):
        torch.manual_seed(7)
        teacher = build_teacher({"dim": semantic.shape[1], "hidden": HIDDEN, "layers": LAYERS,
                                 "n_subtypes": vocab["n_subtypes"], "n_orbits": vocab["n_orbits"]})
        teacher = teacher.to(device)
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
        allowed = vocab["allowed"].to(device)
        data = []
        for family in FAMILIES:
            nodes, targets = prepared[family]["train"]
            data.append((family, torch.from_numpy(nodes).long().to(device),
                         cache[(family, "train", mode)], targets))
        generator = torch.Generator().manual_seed(SEED)
        for step in range(1, STEPS + 1):
            family, index, adjacency, targets = data[step % len(data)]
            kept = np.flatnonzero(targets["subtype"] >= 0)
            draw = torch.randint(0, len(kept), (min(BATCH, len(kept)),), generator=generator).numpy()
            pick = torch.from_numpy(kept[draw]).to(device)
            type_logits, orbit_logits, _ = teacher(semantic[index[pick]], adjacency[pick])
            target_sub = torch.from_numpy(targets["subtype"]).to(device)[pick]
            target_orbit = torch.from_numpy(targets["orbit"]).to(device)[pick]
            loss = F.cross_entropy(type_logits, target_sub.clamp(min=0)) + \
                F.cross_entropy(orbit_logits.reshape(-1, orbit_logits.shape[-1]),
                                target_orbit.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 300 == 0:
                print(f"  {mode} step {step} loss {float(loss.detach()):.4f}", flush=True)

        teacher.eval()
        row = {"variant": mode, "per_family": {}}
        with torch.no_grad():
            for family in FAMILIES:
                nodes, targets = prepared[family]["test"]
                index = torch.from_numpy(nodes).long().to(device)
                adjacency = cache[(family, "test", mode)]
                type_logits, orbit_logits, _ = teacher(semantic[index], adjacency)
                kept = np.flatnonzero(targets["subtype"] >= 0)
                prediction = type_logits.argmax(-1).cpu().numpy()[kept]
                target_sub = targets["subtype"][kept]
                unconditional = orbit_logits.argmax(-1).cpu().numpy()[kept]
                orbit_target = targets["orbit"][kept]
                subtype_ids = torch.from_numpy(target_sub).to(device).clamp(min=0)
                kept_orbit = orbit_logits[torch.from_numpy(kept).to(device)]
                mask = allowed[subtype_ids].unsqueeze(1).expand(-1, kept_orbit.shape[1], -1).clone()
                conditioned = kept_orbit.masked_fill(~mask, float("-inf")).argmax(-1).cpu().numpy()
                row["per_family"][family] = {
                    "subtype_macro_f1": macro_f1(prediction, target_sub, vocab["n_subtypes"]),
                    "orbit_macro_f1": macro_f1(unconditional.reshape(-1), orbit_target.reshape(-1),
                                               vocab["n_orbits"]),
                    "orbit_macro_f1_conditioned": macro_f1(conditioned.reshape(-1),
                                                          orbit_target.reshape(-1), vocab["n_orbits"]),
                    "instances": int(len(kept)),
                    "majority_subtype": float(np.bincount(target_sub).max() / len(kept)),
                    "majority_orbit": float(np.bincount(orbit_target.reshape(-1)).max()
                                            / orbit_target.size),
                }
        row["subtype_macro_f1"] = float(np.mean([v["subtype_macro_f1"] for v in row["per_family"].values()]))
        row["orbit_macro_f1"] = float(np.mean([v["orbit_macro_f1"] for v in row["per_family"].values()]))
        results[mode] = row
        print(json.dumps({"variant": mode, "subtype": row["subtype_macro_f1"],
                          "orbit": row["orbit_macro_f1"]}), flush=True)
        if mode == "directed":
            directed_teacher = teacher

    (HERE / "teacher_controls.json").write_text(json.dumps(results, indent=1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = teacher_config(directed_teacher)
    record = save_teacher(OUT_DIR / "teacher_directed_v1.pth", directed_teacher, config=config,
                          vocab={"mode": "directed", "n_subtypes": vocab["n_subtypes"],
                                 "n_orbits": vocab["n_orbits"]},
                          extra={"pool": "results/research/directed_graphlet_role/instances.npz",
                                 "pool_sha256": sha256(ROLE_DIR / "instances.npz"),
                                 "subtypes": "results/research/directed_graphlet_role/subtypes.json",
                                 "subtypes_sha256": sha256(ROLE_DIR / "subtypes.json"),
                                 "graph_sha256": sha256(REPO / "results" / "research" /
                                                        "behavior_graph" / "graph.npz"),
                                 "seed": SEED, "steps": STEPS, "batch": BATCH,
                                 "hidden": HIDDEN, "layers": LAYERS,
                                 "train_cap": TRAIN_CAP, "test_cap": TEST_CAP,
                                 "torch": torch.__version__})

    # Fixed probe: the batch, its logits and its targets, so a later reload can be
    # checked against the same numbers without re-reading the pool.
    with torch.no_grad():
        probe_nodes, probe_targets = prepared["K3"]["test"]
        index = torch.from_numpy(probe_nodes[:8]).long().to(device)
        type_logits, orbit_logits, hidden = directed_teacher(semantic[index],
                                                             cache[("K3", "test", "directed")][:8])
    torch.save({"family": "K3",
                "nodes": probe_nodes[:8],
                "adjacency": cache[("K3", "test", "directed")][:8].cpu(),
                "type_logits": type_logits.cpu(),
                "orbit_logits": orbit_logits.cpu(),
                "node_hidden": hidden.cpu(),
                "subtype": probe_targets["subtype"][:8],
                "orbit": probe_targets["orbit"][:8]},
               OUT_DIR / "teacher_probe.pt")

    metadata = dict(record)
    metadata["metrics"] = {mode: {"subtype_macro_f1": results[mode]["subtype_macro_f1"],
                                  "orbit_macro_f1": results[mode]["orbit_macro_f1"],
                                  "per_family": results[mode]["per_family"]}
                           for mode in results}
    metadata["probe"] = {"path": str(OUT_DIR / "teacher_probe.pt"),
                         "sha256": sha256(OUT_DIR / "teacher_probe.pt")}
    metadata["git_commit"] = _git_commit()
    (OUT_DIR / "teacher_metadata.json").write_text(json.dumps(metadata, indent=1))
    print(json.dumps({"saved": str(OUT_DIR / "teacher_directed_v1.pth"),
                      "sha256": record["sha256"]}, indent=1))


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


if __name__ == "__main__":
    main()
