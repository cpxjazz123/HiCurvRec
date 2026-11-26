"""Issue #262: actual quantisation margins, endpoint gradients and switch probes.

One process per arm to keep candidate imports isolated. Read-only: no optimizer
step, no SID write, no checkpoint write. All outputs are issue-local.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue262"
RAW = OUT / "_raw"
ASSETS = ROOT / "assets"
EMBEDDING = ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN = ROOT / "results/stage0_build_parquet/train.parquet"
ARMS = {
    100: "TIGER_gnn_attention_sectional_rq",
    101: "TIGER_three_expert_attention_rq",
    102: "TIGER_module_budget_prefix_rq",
    103: "TIGER_periodic_budget_prefix_rq",
    104: "TIGER_role_conditioned_l1_rq",
    105: "TIGER_role_conditioned_l3_rq",
    106: "TIGER_role_conditioned_l2_rq",
    107: "TIGER_topology_residual_l2_rq",
}
EXPECTED = 24587


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_final(arm: str) -> Path:
    paths = sorted((ROOT / "results/stage3_T5Train" / arm).glob("**/test_final.json"))
    if len(paths) != 1:
        raise RuntimeError(f"{arm}: expected one test_final")
    return paths[0]


def load_q(index: int) -> dict:
    path = RAW / "q" / f"{index}.npz"
    with np.load(path, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def quantiles(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    points = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    return {str(p): float(x) for p, x in zip(points, np.quantile(values, points))}


def parameter_checksum(model) -> str:
    h = hashlib.sha256()
    for _, value in sorted(model.state_dict().items()):
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def group_parameters(model) -> dict:
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
    return groups


def flat_grads(output, parameters, retain=True):
    grads = torch.autograd.grad(output, parameters, retain_graph=retain,
                               allow_unused=True, create_graph=False)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, grads)]


def norm(grads) -> float:
    return float(math.sqrt(sum(float((g.double() ** 2).sum()) for g in grads)))


def cosine(left, right) -> float | None:
    ln = norm(left)
    rn = norm(right)
    if ln == 0 or rn == 0:
        return None
    dot = sum(float((a.double() * b.double()).sum()) for a, b in zip(left, right))
    return dot / (ln * rn)


def quant_margins(model, q, sid3, books, z) -> dict:
    """Euclidean actual-codebook margins; native assignment metric is separate."""
    residual = z.copy()
    per_level = []
    for level in range(3):
        distances = np.linalg.norm(residual[:, None, :] - books[level][None, :, :], axis=-1)
        selected = sid3[:, level]
        selected_distance = distances[np.arange(len(sid3)), selected]
        order = np.argsort(distances, axis=1)
        nearest = order[:, 0]
        second = order[:, 1]
        first_distance = distances[np.arange(len(sid3)), nearest]
        second_distance = distances[np.arange(len(sid3)), second]
        selected_rank = (distances < selected_distance[:, None]).sum(axis=1)
        margin = second_distance - first_distance
        selected_margin = second_distance - selected_distance
        normalized = margin / np.maximum(first_distance, 1e-12)
        selected_code = books[level][selected]
        residual_norm = np.linalg.norm(residual - selected_code, axis=1)
        commitment = np.mean((residual - selected_code) ** 2)
        codebook = np.mean((selected_code - residual) ** 2)
        per_level.append({
            "level": level,
            "selected_not_nearest": int((selected != nearest).sum()),
            "selected_top1_fraction": float((selected == nearest).mean()),
            "selected_rank_mean": float(selected_rank.mean()),
            "selected_rank_p90": float(np.quantile(selected_rank, .9)),
            "margin_top1_top2_mean": float(margin.mean()),
            "margin_top1_top2_median": float(np.median(margin)),
            "selected_margin_mean": float(selected_margin.mean()),
            "normalized_gap_mean": float(normalized.mean()),
            "selected_distance_mean": float(selected_distance.mean()),
            "residual_norm_mean": float(residual_norm.mean()),
            "residual_norm_quantiles": quantiles(residual_norm),
            "commitment_mse": float(commitment),
            "codebook_mse": float(codebook),
            "codebook_norm_mean": float(np.linalg.norm(books[level], axis=1).mean()),
        })
        residual = residual - selected_code
    total = np.linalg.norm(z - q, axis=1)
    return {"representation": "hard q; first three SID columns",
            "metric": "Euclidean distance in the actual 32-d codebook space",
            "per_level": per_level,
            "total_quant_error_mean": float(total.mean()),
            "total_quant_error_quantiles": quantiles(total),
            "checkpoint_beta": None,
            "note": "native assignment uses the source's live mixed cost; this is the cross-level actual-code diagnostic"}


def interventions(model, embeddings, sid3) -> dict:
    embeddings = torch.from_numpy(np.array(embeddings, copy=True))
    actual = q if False else None  # noqa: F841

    def export():
        with torch.inference_mode():
            return model.get_indices(embeddings, infer_use_sk=False).cpu().numpy().astype(np.int64)

    original = export()
    result = {"no_intervention": {"shape": list(original.shape),
                                  "exact_sid_match": bool(np.array_equal(original[:, :3], sid3)),
                                  "changed_fraction": float((original[:, :3] != sid3).any(axis=1).mean())}}
    was_behavior = model.behavior_on
    was_curvature = model.curvature_on
    try:
        model.behavior_on = False
        changed = export()
        model.behavior_on = was_behavior
        result["behavior_off"] = {
            "exact_sid_match": bool(np.array_equal(changed[:, :3], sid3)),
            "changed_fraction": float((changed[:, :3] != sid3).any(axis=1).mean()),
            "per_level_change": [float((changed[:, l] != sid3[:, l]).mean()) for l in range(3)],
        }
    except Exception as exc:  # noqa: BLE001
        result["behavior_off"] = {"status": "NA", "reason": repr(exc)}
    try:
        model.curvature_on = False
        changed = export()
        model.curvature_on = was_curvature
        result["curvature_off"] = {
            "exact_sid_match": bool(np.array_equal(changed[:, :3], sid3)),
            "changed_fraction": float((changed[:, :3] != sid3).any(axis=1).mean()),
            "per_level_change": [float((changed[:, l] != sid3[:, l]).mean()) for l in range(3)],
        }
    except Exception as exc:  # noqa: BLE001
        result["curvature_off"] = {"status": "NA", "reason": repr(exc)}
    try:
        original_gate = model.structure.gates_for_items
        def uniform_gate(store, ids, device, chunk=1024, behavior_on=True):
            tokens, weights = original_gate(store, ids, device, chunk=chunk, behavior_on=False)
            return tokens, torch.full_like(weights, 1.0 / weights.shape[-1])
        model.structure.gates_for_items = uniform_gate
        changed = export()
        model.structure.gates_for_items = original_gate
        result["uniform_gate"] = {
            "exact_sid_match": bool(np.array_equal(changed[:, :3], sid3)),
            "changed_fraction": float((changed[:, :3] != sid3).any(axis=1).mean()),
            "per_level_change": [float((changed[:, l] != sid3[:, l]).mean()) for l in range(3)],
        }
    except Exception as exc:  # noqa: BLE001
        try:
            model.structure.gates_for_items = original_gate
        except Exception:
            pass
        result["uniform_gate"] = {"status": "NA", "reason": repr(exc)}
    try:
        model.curvature_on = False
        original_gate = model.structure.gates_for_items
        def uniform_gate2(store, ids, device, chunk=1024, behavior_on=True):
            tokens, weights = original_gate(store, ids, device, chunk=chunk, behavior_on=False)
            return tokens, torch.full_like(weights, 1.0 / weights.shape[-1])
        model.structure.gates_for_items = uniform_gate2
        changed = export()
        result["euclidean_uniform"] = {
            "exact_sid_match": bool(np.array_equal(changed[:, :3], sid3)),
            "changed_fraction": float((changed[:, :3] != sid3).any(axis=1).mean()),
            "per_level_change": [float((changed[:, l] != sid3[:, l]).mean()) for l in range(3)],
        }
    except Exception as exc:  # noqa: BLE001
        result["euclidean_uniform"] = {"status": "NA", "reason": repr(exc)}
    finally:
        model.behavior_on = was_behavior
        model.curvature_on = was_curvature
    return result


def main() -> None:
    index = int(sys.argv[1])
    arm = ARMS[index]
    final = find_final(arm)
    fd = json.loads(final.read_text())
    checkpoint = Path(fd["stage2_checkpoint"])
    sid_path = Path(fd["sid_path"])
    binding = Path(fd["archive_binding"])
    q_blob = load_q(index)
    q = q_blob["q"].astype(np.float64)
    sid3 = q_blob["sid3"].astype(np.int64)
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 3000:
        raise ValueError("not final epoch")
    receipt = json.loads(binding.read_text())
    tree = Path(receipt["source_tree"])
    for name in list(sys.modules):
        if name.split(".")[0] in ("model", "losses", "distances", "train_config", "train_rqvae",
                                  "assignment_floor", "validation_entry"):
            del sys.modules[name]
    sys.path.insert(0, str(tree))
    training = importlib.import_module("train_rqvae")
    if Path(training.__file__).resolve() != tree / "train_rqvae.py":
        raise ValueError("candidate import did not use exact source tree")
    model = training.RQVAE(training._tokenizer_config(), in_dim=768,
                           distance=training.DISTANCE).cpu()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    books = np.stack([model.rq.vq_layers[level].embed.weight.detach().cpu().numpy()
                      for level in range(3)]).astype(np.float64)
    embeddings = np.load(EMBEDDING, allow_pickle=False)
    if embeddings.shape != (EXPECTED, 768) or not np.isfinite(embeddings).all():
        raise ValueError("embedding catalogue mismatch")
    # Encoder output; actual hard-q is the exported SID code sum, not a re-decode.
    z = np.empty((EXPECTED, 32), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, EXPECTED, 1024):
            end = min(start + 1024, EXPECTED)
            z[start:end] = model.encoder(torch.from_numpy(np.array(embeddings[start:end], copy=True))).cpu().numpy()
    if not np.isfinite(z).all():
        raise ValueError("nonfinite encoder output")
    quant = quant_margins(model, q, sid3, books, z)
    quant["checkpoint_beta"] = float(getattr(training, "BETA", float("nan")))
    # Effective rank of z and hard q.
    def effective_rank(values):
        centred = values - values.mean(0)
        eigenvalues = np.linalg.eigvalsh(centred.T @ centred / len(centred)).clip(1e-12)
        return float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())
    quant["effective_rank_z"] = effective_rank(z)
    quant["effective_rank_q"] = effective_rank(q)
    quant["z_sha256"] = hashlib.sha256(z.astype(np.float32).tobytes()).hexdigest()
    quant["q_sha256"] = hashlib.sha256(q.astype(np.float32).tobytes()).hexdigest()

    # Fixed batch: same item identities and seed for every arm.
    import pandas as pd
    frame = pd.read_parquet(TRAIN, columns=["target"])
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    rng = np.random.default_rng(42)
    batch_ids = np.sort(rng.choice(train_ids, size=4096, replace=False))
    batch = torch.from_numpy(np.array(embeddings[batch_ids], copy=True))
    catalogue = torch.from_numpy(np.array(embeddings, copy=True))
    sample = model.structural_sampler().draw(0, 0, 1, track=False)
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
        reconstruction = components["reconstruction"]
        combined = model.combined_loss(original, curvature)
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        g_original = flat_grads(original, parameters)
        g_curvature = flat_grads(curvature, parameters)
        g_combined = flat_grads(combined, parameters)
    finally:
        for layer in model.rq.vq_layers:
            layer._skip_ddp_reduce = False
        model.zero_grad(set_to_none=True)
    groups = group_parameters(model)
    group_report = {}
    for name, members in groups.items():
        ids = {id(p) for p in members}
        indexes = [i for i, p in enumerate(parameters) if id(p) in ids]
        if not indexes:
            group_report[name] = {"status": "NA", "reason": "no trainable parameters"}
            continue
        go = [g_original[i] for i in indexes]
        gc = [g_curvature[i] for i in indexes]
        gb = [g_combined[i] for i in indexes]
        group_report[name] = {
            "original_norm": norm(go), "curvature_norm": norm(gc),
            "combined_norm": norm(gb),
            "original_curvature_cosine": cosine(go, gc),
        }
    after = parameter_checksum(model)
    gradient = {
        "fixed_batch": {"rows": 4096, "seed": 42, "source": str(TRAIN),
                        "item_ids_sha256": hashlib.sha256(batch_ids.tobytes()).hexdigest()},
        "losses": {"original": float(original.detach()), "reconstruction": float(reconstruction.detach()),
                   "curvature": float(curvature.detach()), "combined": float(combined.detach()),
                   "curvature_weight": float(model.curvature_weight)},
        "quant_parts": [{"commitment": float(a.detach()), "codebook": float(b.detach())}
                        for a, b in parts],
        "groups": group_report,
        "parameter_checksum_before": before,
        "parameter_checksum_after": after,
        "parameters_unchanged": before == after,
        "optimizer_steps": 0,
        "scope": "one fixed final-checkpoint forward/autograd; not historical trajectory",
    }
    # Actual interface switches after gradients; restore model state afterwards.
    model.eval()
    switch = interventions(model, embeddings, sid3)
    result = {
        "schema": "genrec.issue262.quant_grad_switch/1",
        "arm": arm, "index": index,
        "identity": {"final": str(final), "final_sha256": sha(final),
                     "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                     "sid": str(sid_path), "sid_sha256": sha(sid_path),
                     "source_tree": str(tree)},
        "quant": quant,
        "gradient": gradient,
        "switch": switch,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"QUANT_GRAD_SWITCH_{index}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"index": index, "arm": arm,
                      "quant_error": quant["total_quant_error_mean"],
                      "losses": gradient["losses"],
                      "unchanged": gradient["parameters_unchanged"],
                      "switch": {k: {kk: vv for kk, vv in v.items() if kk != "per_level_change"}
                                 for k, v in switch.items()}}, indent=2))


if __name__ == "__main__":
    main()
