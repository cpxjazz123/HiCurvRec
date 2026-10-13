from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STAGE2 = ROOT / "stage2_RQ-VAE" / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
if str(STAGE2) not in sys.path:
    sys.path.insert(0, str(STAGE2))

import train_rqvae as trainer
from model import RQVAE
from analysis.supervision_diagnostic.metrics import (
    codebook_occupancy,
    gradient_norms,
    residual_health,
    summarize_representations,
)

EMBEDDING_PATH = ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_PATH = ROOT / "results" / "stage0_build_parquet" / "train.parquet"
VALID_PATH = ROOT / "results" / "stage0_build_parquet" / "valid.parquet"
BASELINE_CHECKPOINT = ROOT / "results" / "stage2_RQ-VAE" / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE" / "resample" / "rqvae_best.pth"
PROBE_STEPS = 200
PROBE_PAIR_BATCH = 256
PROBE_ITEM_BATCH = 1024
PROBE_LR = 1e-3
ACTIVE_BEHAVIOR_WEIGHT = trainer.BEHAVIOR_WEIGHT_MAX


def _validation_triples(valid, train_pairs, n_items):
    positives = []
    for history, target in zip(valid["seen_history"].to_numpy(), valid["target"].to_numpy(dtype=np.int64)):
        if history is not None and len(history):
            positives.append((int(history[-1]), int(target)))
    positives = np.asarray(positives, dtype=np.int64)
    rng = np.random.default_rng(42042)
    known = set(map(tuple, train_pairs.tolist()))
    item_pool = np.unique(train_pairs.reshape(-1))
    sources, targets = positives.T
    negatives = rng.choice(item_pool, size=len(positives), replace=True)
    for idx in range(len(negatives)):
        while negatives[idx] == targets[idx] or (int(sources[idx]), int(negatives[idx])) in known:
            negatives[idx] = rng.choice(item_pool)
    comparison_pairs = np.empty((2 * len(positives), 2), dtype=np.int64)
    comparison_pairs[0::2] = np.column_stack((sources, targets))
    comparison_pairs[1::2] = np.column_stack((sources, negatives))
    labels = np.zeros(2 * len(positives), dtype=np.int64)
    labels[0::2] = 1
    if comparison_pairs.max() >= n_items:
        raise ValueError("Validation item id exceeds embedding rows")
    return positives, comparison_pairs, labels


def _model_from_state(embeddings, state):
    model = RQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1])
    trainer.initialize_tiger_weights(model)
    model.load_state_dict(state)
    return model


def _trace(model, values):
    return trainer._residuals_with_trace(model, values)


def _gradient_report(model, batch, pairs, all_embeddings):
    model.train()
    inputs = all_embeddings[torch.as_tensor(batch, dtype=torch.long, device=all_embeddings.device)]
    pair_ids = torch.as_tensor(pairs, dtype=torch.long, device=all_embeddings.device)
    pair_values = all_embeddings[pair_ids.reshape(-1)]
    residuals = _trace(model, pair_values)
    behavior_loss = trainer.behavior_contrastive_loss(
        residuals, pair_ids[:, 0], pair_ids[:, 1]
    )
    reconstruction, quant_loss, _, _ = model(inputs)
    total_loss, _ = model.compute_loss(inputs, reconstruction, quant_loss)
    total_loss = total_loss + ACTIVE_BEHAVIOR_WEIGHT * behavior_loss
    groups = {"encoder": tuple(model.encoder.parameters())}
    for level, layer in enumerate(model.rq.vq_layers):
        groups[f"codebook_L{level + 1}"] = (layer.get_code_embs(),)

    def measure(loss):
        parameter_norms = gradient_norms(loss, groups)
        residual_gradient = torch.autograd.grad(
            loss, residuals, retain_graph=True, allow_unused=True
        )[0]
        for level in range(residuals.shape[0]):
            gradient = None if residual_gradient is None else residual_gradient[level]
            parameter_norms[f"residual_L{level + 1}"] = (
                float(torch.linalg.vector_norm(gradient.detach().double()).cpu())
                if gradient is not None else 0.0
            )
        return parameter_norms

    return (
        measure(behavior_loss),
        measure(total_loss),
        residuals.detach(),
        float(behavior_loss.detach().cpu()),
        float(total_loss.detach().cpu()),
    )


def _probe_run(train_pairs, all_embeddings, train_items, valid_pairs, valid_labels, weight):
    trainer.set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RQVAE(trainer._tokenizer_config(), in_dim=all_embeddings.shape[1]).to(device)
    trainer.initialize_tiger_weights(model)
    item_ids = torch.as_tensor(train_items, dtype=torch.long)
    init_values = all_embeddings[item_ids]
    model.init_codebook(init_values.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROBE_LR, weight_decay=trainer.WEIGHT_DECAY)
    sampler = torch.Generator().manual_seed(42)
    pair_tensor = torch.as_tensor(train_pairs, dtype=torch.long)
    train_tensor = torch.as_tensor(train_items, dtype=torch.long)
    for _ in range(PROBE_STEPS):
        item_pick = torch.randint(len(train_tensor), (PROBE_ITEM_BATCH,), generator=sampler)
        batch_ids = train_tensor[item_pick]
        pair_pick = torch.randint(len(pair_tensor), (PROBE_PAIR_BATCH,), generator=sampler)
        pairs = pair_tensor[pair_pick]
        batch = all_embeddings[batch_ids].to(device)
        pair_values = all_embeddings[pairs.reshape(-1)].to(device)
        reconstructed, quant_loss, _, _ = model(batch)
        base_loss, _ = model.compute_loss(batch, reconstructed, quant_loss)
        traces = _trace(model, pair_values)
        behavior_loss = trainer.behavior_contrastive_loss(traces, pairs[:, 0].to(device), pairs[:, 1].to(device))
        loss = base_loss + float(weight) * behavior_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        traces = _trace(model, all_embeddings.to(device))
        representations = traces.detach()
        comparison = summarize_representations(representations, valid_pairs, valid_labels)
        tokens = model.get_indices(all_embeddings.to(device), infer_use_sk=False)
        occupancy = codebook_occupancy(tokens.cpu())
        health = residual_health(representations)
    return comparison, occupancy, health


def _verify_capacity_freeze():
    source_path = STAGE2 / "train_rqvae.py"
    source = source_path.read_text(encoding="utf-8")
    resolved = []
    for arm in trainer._ARM_DIR:
        namespace = {"__file__": str(source_path), "__name__": f"freeze_check_{arm}"}
        arm_source = source.replace(
            'EXPERIMENT_ARM = "resample"',
            f'EXPERIMENT_ARM = "{arm}"',
            1,
        )
        exec(compile(arm_source, str(source_path), "exec"), namespace)
        actual = (
            namespace["CODEBOOK_DIM"],
            namespace["CODEBOOK_SIZE"],
            namespace["HIDDEN_SIZES"],
        )
        expected = (32, (256, 256, 256), (512, 256, 128))
        if actual != expected:
            raise AssertionError(f"arm {arm} resolved to {actual}, expected {expected}")
        resolved.append(arm)
    print("Capacity freeze verification (DIM / SIZE / HIDDEN)")
    for arm in ("resample", "crowd_posattract_w1", "crowd_couser_w2"):
        print(f"{arm:<22} 32 / (256,256,256) / (512,256,128)")
    print(f"all non-capacity TIGER behavior arms: {len(resolved)} imports unchanged")

    for arm in trainer._CAPACITY_ARMS:
        namespace = {"__file__": str(source_path), "__name__": f"freeze_reject_{arm}"}
        arm_source = source.replace(
            'EXPERIMENT_ARM = "resample"',
            f'EXPERIMENT_ARM = "{arm}"',
            1,
        )
        try:
            exec(compile(arm_source, str(source_path), "exec"), namespace)
        except RuntimeError as error:
            if "Capacity freeze is active" not in str(error):
                raise
        else:
            raise AssertionError(f"capacity arm {arm} did not fail on import")
    print(f"capacity arm selection: rejected all {len(trainer._CAPACITY_ARMS)} names")


def main():
    _verify_capacity_freeze()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    embeddings_np = np.load(EMBEDDING_PATH).astype(np.float32, copy=False)
    train = pd.read_parquet(TRAIN_PATH)
    valid = pd.read_parquet(VALID_PATH)
    train_pairs = trainer.TransitionPairs(train).pairs.numpy()
    train_items = np.unique(train["target"].to_numpy(dtype=np.int64))
    valid_triples, validation_pairs, validation_labels = _validation_triples(
        valid, train_pairs, len(embeddings_np)
    )
    all_embeddings = torch.from_numpy(embeddings_np)

    with tempfile.NamedTemporaryFile(prefix="supervision-checkpoint-", suffix=".pth", delete=False) as temp:
        staged_checkpoint = Path(temp.name)
    try:
        shutil.copy2(BASELINE_CHECKPOINT, staged_checkpoint)
        checkpoint = torch.load(staged_checkpoint, map_location="cpu")
    finally:
        staged_checkpoint.unlink(missing_ok=True)
    checkpoint_state = checkpoint["state_dict"]
    gradient_model = _model_from_state(embeddings_np, checkpoint_state)
    baseline_positive_pairs = train_pairs[:PROBE_PAIR_BATCH]
    behavior_grads, total_grads, traced, behavior_value, total_value = _gradient_report(
        gradient_model, train_items[:PROBE_ITEM_BATCH], baseline_positive_pairs, all_embeddings
    )
    print("Gradient norms (resample checkpoint; behavior loss / total loss)")
    print("component       behavior_loss       total_loss")
    for key in behavior_grads:
        print(f"{key:<16} {behavior_grads[key]:16.8e} {total_grads[key]:16.8e}")
    print(f"loss_values      behavior={behavior_value:.8e} total={total_value:.8e}")
    print("Checkpoint real-batch residual health")
    for level, item in enumerate(residual_health(traced)):
        print(f"L{level + 1}: {item}")

    results = {}
    for label, weight in (("behavior_on", ACTIVE_BEHAVIOR_WEIGHT), ("behavior_off", 0.0)):
        aucs, occupancy, health = _probe_run(
            train_pairs, all_embeddings, train_items,
            validation_pairs, validation_labels, weight,
        )
        results[label] = (aucs, occupancy, health)
        print(f"{label}: steps={PROBE_STEPS} behavior_weight={weight:.6f}")
        print("  single-level AUC:", {f"L{k + 1}": v for k, v in aucs["single_auc"].items()})
        print("  cumulative AUC:", {f"L{k}": v for k, v in aucs["cumulative_auc"].items()})
        print("  distinct codewords:", {f"L{k + 1}": v for k, v in enumerate(occupancy)})
        print("  residual health:", {f"L{k + 1}": v for k, v in enumerate(health)})
    print("Probe comparison (AUC; occupancy):")
    print("level  behavior_on  behavior_off  on_occupancy  off_occupancy")
    for level in range(3):
        print(f"L{level + 1}     {results['behavior_on'][0]['single_auc'][level]:.6f}      "
              f"{results['behavior_off'][0]['single_auc'][level]:.6f}       "
              f"{results['behavior_on'][1][level]:4d}          {results['behavior_off'][1][level]:4d}")
    print("Cumulative AUC (behavior_on / behavior_off):")
    for count in range(1, 4):
        print(f"L1..L{count}: {results['behavior_on'][0]['cumulative_auc'][count]:.6f} / "
              f"{results['behavior_off'][0]['cumulative_auc'][count]:.6f}")


if __name__ == "__main__":
    main()
