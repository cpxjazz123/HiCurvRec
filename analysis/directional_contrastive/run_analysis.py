from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
import directionality as d


TRAINER_PATH = config.REPO / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/train_rqvae.py"
sys.path.insert(0, str(TRAINER_PATH.parent))
spec = importlib.util.spec_from_file_location("directional_study_trainer", TRAINER_PATH)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
from model import RQVAE


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, Counter):
        return {str(k): int(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_model(checkpoint, device):
    model = RQVAE(trainer._tokenizer_config(), in_dim=768)
    trainer.initialize_tiger_weights(model)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    state = state.get("state_dict", state)
    head_state = {k.split("directional_heads.", 1)[1]: v for k, v in state.items() if k.startswith("directional_heads.")}
    model_state = {k: v for k, v in state.items() if not k.startswith("directional_heads.")}
    model.load_state_dict(model_state)
    model.to(device).eval()
    heads = None
    if head_state:
        heads = trainer.DirectionalHeads()
        heads.load_state_dict(head_state)
        heads.to(device).eval()
    return model, heads




def _per_level_losses(residuals, src_ids, tgt_ids, formula, heads, temperature):
    b = len(src_ids)
    valid = src_ids != tgt_ids
    losses = []
    for level in range(residuals.shape[0]):
        src, cand = residuals[level, :b], residuals[level, b:]
        if formula == "resample":
            logits = -torch.cdist(src, cand) / temperature
        else:
            logits = heads.project_source(src) @ heads.project_target(cand).T / temperature
            if formula == "resample_proj_sym":
                rev = heads.project_source(cand) @ heads.project_target(src).T
                logits = (logits + rev.T) * (0.5 / temperature)
        duplicate = tgt_ids[:, None].eq(tgt_ids[None, :])
        duplicate.fill_diagonal_(False)
        logits = logits.masked_fill(duplicate, -torch.inf)
        if bool(valid.any()):
            loss = F.cross_entropy(logits[valid], torch.arange(b, device=src_ids.device)[valid])
            losses.append(loss)
        else:
            losses.append(residuals.sum() * 0)
    return losses


def _grad_norms(model, heads, residuals, src_ids, tgt_ids, formula, temperature):
    model.zero_grad(set_to_none=True)
    if heads is not None:
        heads.zero_grad(set_to_none=True)
    losses = _per_level_losses(residuals, src_ids, tgt_ids, formula, heads, temperature)
    total = torch.stack(losses).mean()
    total.backward()
    def norm(parameters):
        grads = [p.grad.detach().norm().square() for p in parameters if p.grad is not None]
        return float(torch.stack(grads).sum().sqrt().cpu()) if grads else 0.0
    return {"encoder": norm(model.encoder.parameters()), "projection_heads": norm(heads.parameters()) if heads is not None else None,
            "contrastive_loss": float(total.detach().cpu())}


def _sid_metrics(path, n_items):
    sid = np.load(path, allow_pickle=False)
    sid = np.asarray(sid, dtype=np.int64)
    raw = sid[:, :3]
    out = {"shape": list(sid.shape), "per_level": {}}
    for level in range(3):
        counts = np.bincount(raw[:, level])
        p = counts[counts > 0] / len(raw)
        out["per_level"][config.LEVEL_NAMES[level]] = {
            "distinct_codes": int(np.count_nonzero(counts)), "entropy_nats": float(-(p * np.log(p)).sum()),
            "entropy_bits": float(-(p * np.log2(p)).sum())}
    def collisions(a):
        _, counts = np.unique(a, axis=0, return_counts=True)
        return {"distinct_sids": int(len(counts)), "collision_rate": float(1 - len(counts) / len(a))}
    out["collision_before_extension"] = collisions(raw)
    out["collision_after_extension"] = collisions(sid)
    out["n_expected_items"] = int(n_items)
    return out


def _behavior_sid_metrics(sid, valid_pairs, batch_size):
    levels = np.asarray(sid[:, :3], dtype=np.int64)
    positives = np.asarray([(int(s), int(t)) for s, t in valid_pairs if s != t and s < len(levels) and t < len(levels)], dtype=np.int64)
    positive_keys = np.unique(positives[:, 0] * len(levels) + positives[:, 1]) if len(positives) else np.empty(0, dtype=np.int64)
    names = ("L1", "L1+L2", "L1+L2+L3")
    pos_counts = np.zeros(3, dtype=np.int64)
    if len(positives):
        for depth in (1, 2, 3):
            pos_counts[depth - 1] = np.all(levels[positives[:, 0], :depth] == levels[positives[:, 1], :depth], axis=1).sum()
    neg_shared = np.zeros(3, dtype=np.int64)
    neg_total = 0
    for start in range(0, len(valid_pairs), batch_size):
        batch = np.asarray(valid_pairs[start:start + batch_size], dtype=np.int64)
        batch = batch[batch[:, 0] != batch[:, 1]]
        if not len(batch):
            continue
        sources, true_targets = batch[:, 0], batch[:, 1]
        candidates = np.asarray(valid_pairs[start:start + batch_size], dtype=np.int64)[:, 1]
        candidate_keys = sources[:, None] * len(levels) + candidates[None, :]
        valid = (candidates[None, :] != sources[:, None]) & (candidates[None, :] != true_targets[:, None])
        if len(positive_keys):
            valid &= ~np.isin(candidate_keys, positive_keys)
        if not valid.any():
            continue
        neg_total += int(valid.sum())
        for depth in (1, 2, 3):
            shared = np.all(levels[sources, None, :depth] == levels[candidates[None, :], :depth], axis=2)
            neg_shared[depth - 1] += int((shared & valid).sum())
    pos_rates = pos_counts / len(positives) if len(positives) else np.full(3, np.nan)
    neg_rates = neg_shared / neg_total if neg_total else np.full(3, np.nan)
    metrics = {}
    for i, name in enumerate(names):
        p, n = float(pos_rates[i]), float(neg_rates[i])
        metrics[name] = {"positive_share_rate": p, "negative_share_rate": n, "gap": p - n,
                         "er": p / n if n > 0 else None,
                         "er_undefined_reason": None if n > 0 else "negative sharing rate is zero"}
    return metrics, int(len(positives)), int(neg_total)


def _run_arm(arm_key, arm, valid_pairs, embeddings, device, graph):
    arm_dir = config.STAGE2 / arm["name"]
    ckpt = arm_dir / "rqvae_best.pth"
    sid_file = arm_dir / "sids_for_hgrec.npy"
    if not ckpt.is_file() or not sid_file.is_file():
        return None
    model, heads = _load_model(ckpt, device)
    formula = arm["formula"]
    if formula != "resample" and heads is None:
        raise RuntimeError(f"{arm_key}: directional checkpoint is missing directional_heads.*")
    all_emb = torch.as_tensor(np.array(embeddings, copy=True), dtype=torch.float32, device=device)
    # Per-item residual trace from this arm's own trained checkpoint.
    residual_chunks = []
    with torch.no_grad():
        for start in range(0, len(all_emb), 4096):
            residual_chunks.append(trainer._residuals_with_trace(model, all_emb[start:start + 4096]).cpu())
    residuals = torch.cat(residual_chunks, dim=1).to(device)
    categories = d.classify_pairs(graph, valid_pairs)
    levels_by_kind = {}
    for kind in ("REVERSE-OBSERVED", "REVERSE-UNOBSERVED", "SELF"):
        chosen = [pair for pair, label in zip(valid_pairs, categories) if label == kind]
        if not chosen:
            levels_by_kind[kind] = []
            continue
        group = np.asarray(chosen, dtype=np.int64)
        levels_by_kind[kind] = d.evaluate_residual_scores(
            residuals, group[:, 0], group[:, 1], formula, heads,
            trainer.BEHAVIOR_TEMPERATURE, config.BATCH_SIZE, include_self=(kind == "SELF"))
    stratified = {}
    for name, group_pairs in d.stratify_pairs(graph, valid_pairs).items():
        group = np.asarray(group_pairs, dtype=np.int64)
        stratified[name] = d.evaluate_residual_scores(
            residuals, group[:, 0], group[:, 1], formula, heads,
            trainer.BEHAVIOR_TEMPERATURE, config.BATCH_SIZE)
    # A single fixed real batch is used for the gradient-flow measurement.
    real_batch = next((valid_pairs[i:i + config.BATCH_SIZE] for i in range(0, len(valid_pairs), config.BATCH_SIZE)
                       if len(valid_pairs[i:i + config.BATCH_SIZE]) > 1 and any(a != b for a, b in valid_pairs[i:i + config.BATCH_SIZE])), None)
    if real_batch is None:
        raise RuntimeError("validation split contains no non-self pair batch")
    pair_arr = np.asarray(real_batch, dtype=np.int64)
    item_ids = np.concatenate([pair_arr[:, 0], pair_arr[:, 1]])
    real_embeddings = all_emb[torch.as_tensor(item_ids, device=device)]
    trace = trainer._residuals_with_trace(model, real_embeddings)
    source_ids, target_ids = (torch.as_tensor(pair_arr[:, i], device=device) for i in (0, 1))
    grad = _grad_norms(model, heads, trace, source_ids, target_ids, formula, trainer.BEHAVIOR_TEMPERATURE)
    losses = _per_level_losses(trace, source_ids, target_ids, formula, heads, trainer.BEHAVIOR_TEMPERATURE)
    per_level_auxiliary = []
    for level, loss in enumerate(losses):
        per_level_auxiliary.append({"level": config.LEVEL_NAMES[level],
                                    "final_checkpoint_contrastive_loss": float(loss.detach().cpu())})
    # Score directionality over vector-quantized code embeddings, still split by
    # reverse-observation status and self transitions.
    quantized = []
    with torch.no_grad():
        for start in range(0, len(all_emb), 4096):
            _, q, _, _ = d.quantized_residuals(model, all_emb[start:start + 4096])
            quantized.append(q.detach().cpu())
    q = torch.cat(quantized, dim=1).to(device)
    code_gap_by_kind = {}
    for kind in ("REVERSE-OBSERVED", "REVERSE-UNOBSERVED", "SELF"):
        chosen = [pair for pair, label in zip(valid_pairs, categories) if label == kind]
        if not chosen:
            code_gap_by_kind[kind] = []
            continue
        group = np.asarray(chosen, dtype=np.int64)
        code_scores = d.evaluate_residual_scores(
            q, group[:, 0], group[:, 1], formula, heads,
            trainer.BEHAVIOR_TEMPERATURE, config.BATCH_SIZE, include_self=(kind == "SELF"))
        code_gap_by_kind[kind] = [x["direction_gap"] for x in code_scores]
    sid = np.load(sid_file, allow_pickle=False)
    sid_stats = _sid_metrics(sid_file, len(embeddings))
    share, n_pos, n_neg = _behavior_sid_metrics(sid, valid_pairs, config.BATCH_SIZE)
    sid_stats["behavior_enrichment_and_sharing"] = share
    sid_stats["behavior_pair_counts"] = {"positive": n_pos, "non_successor_comparison": n_neg}
    sid_stats["reconstruction"] = _reconstruction_metrics(model, all_emb, sid)
    return {"arm_name": arm["name"], "score_formula": formula, "levels_by_pair_kind": levels_by_kind,
            "degree_stratified_validation_levels": stratified,
            "per_level_auxiliary": per_level_auxiliary, "gradient_norms_fixed_batch": grad,
            "quantized_code_direction_gap_by_pair_kind": code_gap_by_kind,
            "semantic_ids": sid_stats}


def _reconstruction_metrics(model, embeddings, sid):
    codes = torch.as_tensor(np.asarray(sid[:, :3], dtype=np.int64), device=embeddings.device)
    code_vectors = [layer.get_code_embs()[codes[:, level]] for level, layer in enumerate(model.rq.vq_layers)]
    qsum = torch.stack(code_vectors).sum(dim=0)
    with torch.no_grad():
        encoded = model.encoder(embeddings)
        decoded = model.decoder(qsum)
        quantized_mse = F.mse_loss(qsum, encoded)
        decoded_mse = F.mse_loss(decoded, embeddings)
    return {"quantized_vs_encoder_mse": float(quantized_mse.cpu()),
            "decoded_vs_input_mse": float(decoded_mse.cpu())}


def main():
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train = pd.read_parquet(config.TRAIN_PARQUET, columns=["seen_history", "target"])
    valid = pd.read_parquet(config.VALID_PARQUET, columns=["seen_history", "target"])
    graph = d.build_transition_graph(train)
    valid_pairs = d.pairs_from_frame(valid)
    embeddings = np.load(config.EMBEDDING_FILE, mmap_mode="r")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"warning": "A missing reverse edge is NOT evidence that the reverse relation is false; it is only evidence that it is unobserved.",
               "data": {"train_pair_rows": len(d.pairs_from_frame(train)), "validation_pair_rows": len(valid_pairs), "validation_source": str(config.VALID_PARQUET), "test_split_used": False},
               "directed_train_graph": {"edge_observation_counts": {f"{s}->{t}": int(n) for (s, t), n in graph["edge_counts"].items()},
                   "reverse_edge_observed": {f"{s}->{t}": bool(value) for (s, t), value in graph["reverse_observed"].items()},
                   "out_degree_distinct_neighbors": graph["out_degree"], "in_degree_distinct_neighbors": graph["in_degree"],
                   "out_strength_observations": graph["out_strength"], "in_strength_observations": graph["in_strength"],
                   "train_frequency_definition": "number of train.parquet target occurrences per item",
                   "train_frequency": graph["train_frequency"]}}
    categories = d.classify_pairs(graph, valid_pairs)
    results["validation_pair_kinds"] = dict(Counter(categories))
    results["validation_degree_observation_strata"] = {name: {"count": len(pairs)} for name, pairs in d.stratify_pairs(graph, valid_pairs).items()}
    results["arms"] = {}
    results["missing_arms"] = []
    for key, arm in config.ARMS.items():
        if not (config.STAGE2 / arm["name"] / "rqvae_best.pth").is_file() or not (config.STAGE2 / arm["name"] / "sids_for_hgrec.npy").is_file():
            results["missing_arms"].append(key)
            print(f"MISSING arm {key}: {arm['name']}")
            continue
        results["arms"][key] = _run_arm(key, arm, valid_pairs, embeddings, device, graph)
        print(f"ANALYZED arm {key}: {arm['name']}")
    output = config.OUT_DIR / "directional_contrastive_results.json"
    output.write_text(json.dumps(_safe(results), indent=2, sort_keys=True) + "\n")
    print(f"RESULTS_JSON {output}")
    print("PER_LEVEL_DIRECTION_GAP")
    for key, arm_result in results["arms"].items():
        for kind, levels in arm_result["levels_by_pair_kind"].items():
            for level_name, level in zip(config.LEVEL_NAMES, levels):
                gap = level["direction_gap"]
                print(f"{key}\t{kind}\t{level_name}\tmean={gap['mean']:.9f}\tmedian={gap['median']:.9f}")
    print("MISSING_ARMS " + (", ".join(results["missing_arms"]) if results["missing_arms"] else "none"))


if __name__ == "__main__":
    main()
