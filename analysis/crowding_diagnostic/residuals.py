import hashlib
import importlib.util
import json
import sys

import numpy as np

from config import ARMS, EMBEDDINGS, LEVELS, OUT_DIR, ROOT


def residual_trace_from_model(model, embeddings, trainer=None):
    import torch
    trainer = trainer or _trainer_module()
    x = torch.tensor(np.asarray(embeddings), dtype=torch.float32)
    with torch.no_grad():
        trace = trainer._residuals_with_trace(model, x)
    return trace.cpu().numpy().transpose(1, 0, 2)


def norm_statistics(residuals):
    values = np.linalg.norm(np.asarray(residuals, dtype=np.float64), axis=-1)
    keys = ("mean", "median", "std", "min", "max", "p01", "p05", "p25", "p75", "p95", "p99")
    rows = []
    for level in range(values.shape[1]):
        row = {"level": LEVELS[level], "mean": float(values[:, level].mean()),
               "median": float(np.median(values[:, level])), "std": float(values[:, level].std()),
               "min": float(values[:, level].min()), "max": float(values[:, level].max())}
        for p in (1, 5, 25, 75, 95, 99):
            row[f"p{p:02d}"] = float(np.percentile(values[:, level], p))
        rows.append({key: row[key] for key in ("level", *keys)})
    return rows


def _trainer_module():
    path = ROOT / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/train_rqvae.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("crowding_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract(model_label):
    import torch
    trainer = _trainer_module()
    from model import RQVAE
    arm = ARMS[model_label]
    actual_hash = hashlib.sha256(arm["checkpoint"].read_bytes()).hexdigest()
    if actual_hash != arm["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch for {model_label}: {actual_hash}")
    model = RQVAE(trainer._tokenizer_config(), in_dim=768)
    trainer.initialize_tiger_weights(model)
    checkpoint = torch.load(arm["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    embeddings = np.load(EMBEDDINGS, mmap_mode="r")
    ids = np.array(sorted(map(int, json.loads(arm["sid_json"].read_text()).keys())), dtype=np.int64)
    if embeddings.shape[0] == len(ids):
        selected = np.asarray(embeddings, dtype=np.float32)
    else:
        selected = np.asarray(embeddings[ids], dtype=np.float32)
    if len(selected) != 24587:
        raise ValueError(f"Expected 24587 aligned item embeddings, got {len(selected)}")
    trace = residual_trace_from_model(model, selected, trainer)
    with torch.no_grad():
        tensor = torch.tensor(selected, dtype=torch.float32)
        encoded = model.encoder(tensor)
        _, _, _, codes = model.rq(encoded)
        codes = codes.cpu().numpy().astype(np.int64)
        recon = np.zeros_like(encoded.cpu().numpy())
        residual = encoded
        for layer in model.rq.vq_layers:
            quantized, _, _, _ = layer(residual)
            recon += quantized.cpu().numpy()
            residual = residual - quantized
    sid = trainer._extend_collisions(codes, list(model.config.codebook_size))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"residuals_{model_label}.npz"
    np.savez_compressed(output, item_ids=ids, residuals=trace.astype(np.float32), codes=codes,
                        cumulative_reconstruction=recon.astype(np.float32), sid=sid,
                        checkpoint_sha256=np.asarray(actual_hash))
    result = {"model": model_label, "checkpoint": str(arm["checkpoint"]), "checkpoint_sha256": actual_hash,
              "residual_file": str(output), "n_items": len(trace), "residual_norms": norm_statistics(trace),
              "interpretation": "A smaller residual norm is NOT by itself evidence of crowding."}
    return result
