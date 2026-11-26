"""Issue #266: measure the frozen endpoint C and G1 on R2 arms 108/109.

Read-only diagnostic. One fresh process per arm; no optimizer is constructed,
no parameter is written, no SID is rewritten. The forward/autograd definitions
copy issue262/quant_grad_switch.py; the only difference is that q preparation is
unnecessary for C/G1 and is therefore absent.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EMBEDDING = ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN = ROOT / "results/stage0_build_parquet/train.parquet"
EXPECTED = 24587
ARMS = {
    108: "TIGER_frozen_topology_gnn_rq",
    109: "TIGER_euclid_local_curve_supervision_rq",
}
EXPECTED_BATCH_SHA = "251dced08f5c42703e2ce66b3c56924e91d14200a15a3771f45c8d8c66f1c457"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(values, dtype=None) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def tree_sha(root: Path) -> str:
    """Hash source code/config, not pycache or generated result files."""
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".json", ".yaml", ".yml"}:
            continue
        relative = path.relative_to(root).as_posix()
        h.update(relative.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def parameter_checksum(model) -> str:
    h = hashlib.sha256()
    for _, value in sorted(model.state_dict().items()):
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def flat_grads(output, parameters):
    grads = torch.autograd.grad(output, parameters, retain_graph=True,
                                allow_unused=True, create_graph=False)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, grads)]


def norm(grads) -> float:
    return float(math.sqrt(sum(float((g.double() ** 2).sum()) for g in grads)))


def sample_record(sample) -> dict:
    return {
        "centre": int(sample["centre"]),
        "ids": {"n": int(len(sample["ids"])),
                "sha256": array_sha(sample["ids"], np.int64)},
        "triples_global": {"shape": list(np.asarray(sample["triples_global"]).shape),
                           "sha256": array_sha(sample["triples_global"], np.int64)},
        "triples": {"shape": list(np.asarray(sample["triples"]).shape),
                    "sha256": array_sha(sample["triples"], np.int64)},
        "references_global": {"n": int(len(sample["references_global"])),
                              "sha256": array_sha(sample["references_global"], np.int64)},
        "references": {"n": int(len(sample["references"])),
                       "sha256": array_sha(sample["references"], np.int64)},
        "target": {"n": int(len(sample["target"])),
                   "sha256": array_sha(sample["target"]),
                   "dtype": str(np.asarray(sample["target"]).dtype)},
        "row_ids": {"n": int(len(sample["row_ids"])),
                    "sha256": array_sha(sample["row_ids"], np.int64)},
        "sampler_counts": {
            "reference_population": int(sample["reference_population"]),
            "centre_population": int(sample["centre_population"]),
            "triple_population": int(sample["triple_population"]),
            "global_step": int(sample["global_step"]),
            "rank": int(sample["rank"]),
            "world_size": int(sample["world_size"]),
        },
    }


def main() -> None:
    index = int(sys.argv[1])
    arm = ARMS[index]
    issue_dir = HERE
    bundle = issue_dir / "_r2" / str(index)
    final = sorted((ROOT / "results/stage3_T5Train" / arm).glob("**/test_final.json"))
    if len(final) != 1:
        raise RuntimeError(f"{arm}: expected exactly one test_final.json, got {len(final)}")
    final = final[0]
    fd = json.loads(final.read_text())
    checkpoint = Path(fd["stage2_checkpoint"])
    sid_path = Path(fd["sid_path"])
    binding_path = Path(fd["archive_binding"])
    if not checkpoint.is_file() or not sid_path.is_file() or not binding_path.is_file():
        raise FileNotFoundError("a bound checkpoint/SID/binding is missing")
    binding = json.loads(binding_path.read_text())
    tree = Path(binding["source_tree"])
    if not (tree / "train_rqvae.py").is_file():
        raise FileNotFoundError("source tree is missing train_rqvae.py")
    if sha(checkpoint) != binding["checkpoint"]["sha256"]:
        raise ValueError("checkpoint hash does not match archive_binding")
    if sha(sid_path) != binding["sid"]["sha256"]:
        raise ValueError("SID hash does not match archive_binding")
    recall_path = bundle / "recall.json"
    recall = json.loads(recall_path.read_text())
    if float(recall["test_recall@10"]) != float(fd["test_recall@10"]):
        raise ValueError("archived recall and local test_final disagree")
    if recall["stage2_checkpoint_sha256"] != binding["checkpoint"]["sha256"]:
        raise ValueError("archived recall checkpoint identity disagrees")
    if recall["sid_sha256"] != binding["sid"]["sha256"]:
        raise ValueError("archived recall SID identity disagrees")

    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 3000:
        raise ValueError("not a final epoch checkpoint")
    for name in list(sys.modules):
        if name.split(".")[0] in ("model", "losses", "distances", "train_config", "train_rqvae",
                                  "assignment_floor", "validation_entry"):
            del sys.modules[name]
    sys.path.insert(0, str(tree))
    training = importlib.import_module("train_rqvae")
    if Path(training.__file__).resolve() != tree / "train_rqvae.py":
        raise ValueError("candidate import did not use the exact source tree")
    model = training.RQVAE(training._tokenizer_config(), in_dim=768,
                           distance=training.DISTANCE).cpu()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    embeddings = np.load(EMBEDDING, allow_pickle=False)
    if embeddings.shape != (EXPECTED, 768) or not np.isfinite(embeddings).all():
        raise ValueError("embedding catalogue mismatch")
    frame = pd.read_parquet(TRAIN, columns=["target"])
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    rng = np.random.default_rng(42)
    batch_ids = np.sort(rng.choice(train_ids, size=4096, replace=False))
    batch_sha = array_sha(batch_ids, np.int64)
    if batch_sha != EXPECTED_BATCH_SHA:
        raise ValueError("fixed batch identity differs from issue262")
    batch = torch.from_numpy(np.array(embeddings[batch_ids], copy=True))
    catalogue = torch.from_numpy(np.array(embeddings, copy=True))
    sample = model.structural_sampler().draw(0, 0, 1, track=False)
    sample_hash = sample_record(sample)
    before = parameter_checksum(model)
    model.train()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    try:
        model.zero_grad(set_to_none=True)
        reconstructed, parts, _, _, curvature = model(
            batch, torch.from_numpy(batch_ids.astype(np.int64)),
            catalogue_embeddings=catalogue, structural_sample=sample)
        original, components = training.CRITERION(reconstructed, batch, parts)
        combined = model.combined_loss(original, curvature)
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        g_original = flat_grads(original, parameters)
        g_curvature = flat_grads(curvature, parameters)
        g_combined = flat_grads(combined, parameters)
    finally:
        for layer in model.rq.vq_layers:
            layer._skip_ddp_reduce = False
        model.zero_grad(set_to_none=True)
    after = parameter_checksum(model)

    groups = {}
    groups["encoder"] = tuple(model.encoder.parameters())
    groups["decoder"] = tuple(model.decoder.parameters())
    structure = []
    gate = []
    for name, parameter in model.named_parameters():
        if name.startswith("structure."):
            structure.append(parameter)
            if "attention" in name or "gate" in name:
                gate.append(parameter)
    groups["GNN"] = tuple(structure)
    groups["gate"] = tuple(gate)
    for level, layer in enumerate(model.rq.vq_layers):
        groups[f"codebook{level}"] = tuple(layer.parameters())

    group_report = {}
    for name, members in groups.items():
        member_ids = {id(parameter) for parameter in members}
        indexes = [i for i, parameter in enumerate(parameters) if id(parameter) in member_ids]
        if not indexes:
            group_report[name] = {"status": "NA", "reason": "no trainable parameters"}
            continue
        go = [g_original[i] for i in indexes]
        gc = [g_curvature[i] for i in indexes]
        gb = [g_combined[i] for i in indexes]
        group_report[name] = {
            "parameters": len(indexes),
            "shapes": [list(g.shape) for g in gc],
            "curvature_nonzero": int(sum(float((g != 0).sum()) for g in gc)),
            "curvature_none": int(sum(1 for g in gc if g is None)),
            "original_norm": norm(go),
            "curvature_norm": norm(gc),
            "combined_norm": norm(gb),
        }

    codebook0_ids = {id(parameter) for parameter in groups["codebook0"]}
    codebook0_indexes = [i for i, parameter in enumerate(parameters) if id(parameter) in codebook0_ids]
    if not codebook0_indexes:
        raise RuntimeError("empty L1 codebook parameter group")
    g1 = norm([g_curvature[i] for i in codebook0_indexes])
    if not math.isfinite(g1) or g1 < 0:
        raise ValueError("invalid G1")
    c = float(curvature.detach())
    if not math.isfinite(c):
        raise ValueError("invalid C")

    result = {
        "schema": "genrec.issue266.endpoint/1",
        "issue": 266,
        "index": index,
        "arm": arm,
        "measurement": {
            "scope": "one fixed final-checkpoint forward/autograd; read-only",
            "C_name": "grad_loss_curvature",
            "G1_name": "grad_curvature_codebook0",
            "C": c,
            "G1": g1,
            "C_unweighted": True,
            "G1_unweighted": True,
            "curvature_weight": float(getattr(model, "curvature_weight", float("nan"))),
            "curvature_on": bool(getattr(model, "curvature_on", True)),
            "behavior_on": bool(getattr(model, "behavior_on", True)),
            "loss_definition": "fifth return of model(...) with catalogue_embeddings + structural_sample; not combined_loss",
            "gradient_definition": "torch.autograd.grad(curvature, all requires_grad parameters, allow_unused=True); G1 over rq.vq_layers[0].parameters()",
            "optimizer_steps": 0,
            "parameters_unchanged": before == after,
            "parameter_checksum_before": before,
            "parameter_checksum_after": after,
        },
        "fixed_batch": {
            "rows": 4096,
            "seed": 42,
            "source": str(TRAIN),
            "item_ids_sha256": batch_sha,
            "expected_sha256": EXPECTED_BATCH_SHA,
            "match": batch_sha == EXPECTED_BATCH_SHA,
        },
        "structural_sample": sample_hash,
        "losses": {
            "original": float(original.detach()),
            "reconstruction": float(components["reconstruction"].detach()),
            "curvature": c,
            "combined": float(combined.detach()),
        },
        "groups": group_report,
        "g1_parameters": {
            "count": len(codebook0_indexes),
            "shapes": [list(g_curvature[i].shape) for i in codebook0_indexes],
            "none_count": int(sum(1 for i in codebook0_indexes if g_curvature[i] is None)),
        },
        "identity": {
            "final": str(final), "final_sha256": sha(final),
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "sid": str(sid_path), "sid_sha256": sha(sid_path),
            "binding": str(binding_path), "binding_sha256": sha(binding_path),
            "source_tree": str(tree), "source_tree_sha256": tree_sha(tree),
            "recall": str(recall_path), "recall_sha256": sha(recall_path),
            "recall_test_recall@10": float(recall["test_recall@10"]),
            "recall_test_recall@5": float(recall["test_recall@5"]),
            "recall_test_ndcg@10": float(recall["test_ndcg@10"]),
            "n_eval": int(recall["n_eval"]),
            "token_cache": str(ROOT / "results/stage2_RQ-VAE/TIGER_frozen_topology_gnn_rq/token_cache.npz"),
            "token_cache_sha256": sha(ROOT / "results/stage2_RQ-VAE/TIGER_frozen_topology_gnn_rq/token_cache.npz"),
        },
    }
    if not result["measurement"]["parameters_unchanged"]:
        raise RuntimeError("parameters changed during diagnostic")
    out = issue_dir / f"ENDPOINT_{index}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"index": index, "arm": arm, "C": c, "G1": g1,
                      "unchanged": before == after,
                      "batch_sha": batch_sha,
                      "sample_ids_sha": sample_hash["ids"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
