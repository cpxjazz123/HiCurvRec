from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import torch

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
ARM_ROOT = ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
EMBEDDINGS_PATH = ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
OUTPUT_PATH = ROOT / "results/margin_analysis/information_probe.json"
ARMS = ("resample", "B_E", "LETTER", "MARGIN", "DIGER", "B_E_nocb")
CODEBOOK_SIZE = 256
CHUNK_ROWS = 1024

sys.path.insert(0, str(ROOT / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"))
sys.path.insert(0, str(ROOT / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/scripts"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis/positive_negative_competition"))
sys.path.insert(0, str(ROOT / "analysis/crowding_diagnostic"))

import train_rqvae
import quant_rank_loss
import competition
import screen_competition
from model import RQVAE


def _auc(vectors: np.ndarray, pairs: dict[str, np.ndarray]) -> float:
    anchors = np.asarray(pairs["anchors"], dtype=np.int64).reshape(-1)
    positives = np.asarray(pairs["positives"], dtype=np.int64)
    negatives = np.asarray(pairs["negatives"], dtype=np.int64)
    if positives.ndim == 1:
        positives = positives[:, None]
    if negatives.ndim == 1:
        negatives = negatives[:, None]
    weighted_auc = 0.0
    weighted_triples = 0
    n_rows = len(anchors)
    for start in range(0, n_rows, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, n_rows)
        anchor_vectors = vectors[anchors[start:stop]]
        positive_distance = np.linalg.norm(
            anchor_vectors[:, None, :] - vectors[positives[start:stop]], axis=2
        )
        negative_distance = np.linalg.norm(
            anchor_vectors[:, None, :] - vectors[negatives[start:stop]], axis=2
        )
        triple_count = positive_distance.size
        weighted_auc += competition.behavioral_auc(positive_distance, negative_distance) * triple_count
        weighted_triples += triple_count
    return weighted_auc / weighted_triples


def _codebook_metrics(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        item_sids = json.load(stream)
    sids = np.asarray(list(item_sids.values()), dtype=np.int64)
    if sids.ndim != 2 or sids.shape[1] < 3:
        raise ValueError(f"Expected at least three SID levels in {path}")
    distinct = [int(np.unique(sids[:, level]).size) for level in range(3)]
    counts = np.bincount(sids[:, 0], minlength=CODEBOOK_SIZE).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(CODEBOOK_SIZE))
    return {
        "distinct_codes_per_level": distinct,
        "l1_top1_share": float(counts.max() / counts.sum()),
        "l1_normalized_entropy": entropy,
        "three_digit_uniqueness": int(np.unique(sids[:, :3], axis=0).shape[0]),
    }


def _measure_arm(arm: str, embeddings: torch.Tensor, pairs: dict[str, np.ndarray]) -> dict[str, object]:
    source_checkpoint = ARM_ROOT / arm / "rqvae_best.pth"
    with tempfile.NamedTemporaryFile(prefix=f"information_probe_{arm}_", suffix=".pth", dir="/tmp", delete=False) as staged:
        staged_checkpoint = Path(staged.name)
    try:
        shutil.copyfile(source_checkpoint, staged_checkpoint)
        model = RQVAE(train_rqvae._tokenizer_config(), in_dim=768)
        state = torch.load(staged_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("state_dict", state), strict=False)
        model.eval()
        with torch.no_grad():
            residuals = train_rqvae._residuals_with_trace(model, embeddings)
            token_ids = model.get_indices(embeddings).t().contiguous()
            quantized = quant_rank_loss.quantized_representation(
                residuals, token_ids, model.rq.vq_layers
            ).detach()
        residual_np = residuals.cpu().numpy()
        continuous_auc = _auc(residual_np[0], pairs)
        residual_aucs = [_auc(residual_np[level], pairs) for level in range(3)]
        quantized_auc = _auc(quantized.cpu().numpy(), pairs)
        result = {
            "arm": arm,
            "continuous_encoder_auc": continuous_auc,
            "residual_l1_auc": residual_aucs[0],
            "residual_l2_auc": residual_aucs[1],
            "residual_l3_auc": residual_aucs[2],
            "cumulative_quantized_auc": quantized_auc,
            "codebook": _codebook_metrics(ARM_ROOT / arm / "item_sids.json"),
        }
        del model, state, residuals, token_ids, quantized
        return result
    finally:
        staged_checkpoint.unlink(missing_ok=True)


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {OUTPUT_PATH}")
    pairs = screen_competition._load_pairs()
    embeddings = torch.from_numpy(np.load(EMBEDDINGS_PATH).astype(np.float32, copy=False))
    rows = [_measure_arm(arm, embeddings, pairs) for arm in ARMS]
    result = {
        "auc_function": "competition.behavioral_auc -> crowding_diagnostic.distances.auc_neg_farther_than_pos",
        "auc_convention": "P(d_negative > d_positive) + 0.5 * P(tie)",
        "triples": int(np.asarray(pairs["anchors"]).size),
        "trace_context": "torch.no_grad()",
        "rows": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    headers = (
        "arm", "continuous_auc", "residual_l1_auc", "residual_l2_auc", "residual_l3_auc",
        "cumulative_quantized_auc", "distinct_l1/256", "distinct_l2/256", "distinct_l3/256",
        "l1_top1_share", "l1_normalized_entropy", "three_digit_uniqueness",
    )
    print("\t".join(headers))
    for row in rows:
        codebook = row["codebook"]
        values = [
            row["arm"],
            *(f"{row[key]:.9f}" for key in (
                "continuous_encoder_auc", "residual_l1_auc", "residual_l2_auc", "residual_l3_auc",
                "cumulative_quantized_auc",
            )),
            *(str(value) for value in codebook["distinct_codes_per_level"]),
            f"{codebook['l1_top1_share']:.9f}",
            f"{codebook['l1_normalized_entropy']:.9f}",
            str(codebook["three_digit_uniqueness"]),
        ]
        print("\t".join(values))


if __name__ == "__main__":
    main()
