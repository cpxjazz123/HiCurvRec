"""Issue #71 §4 — byte-level proof that each exported SID really came from its checkpoint.

Load the saved ``rqvae_best.pth`` state dict into a freshly constructed ``RQVAE``, re-run the
trainer's own export path (``get_indices`` -> ``_extend_collisions``) on the pinned stage-1
embeddings, and compare against the committed ``item_sids.json``.

Numerical configuration is part of the contract
-----------------------------------------------
Every trainer sets ``torch.set_float32_matmul_precision("high")`` at module scope, which turns
on TF32 for fp32 matmuls. A handful of items sit exactly on a code-assignment boundary, so
reproducing the export under full fp32 precision or on CPU changes ~0.4% of the tokens even
though the weights and the input are bit-identical. This script therefore evaluates the export
under the **production** configuration (CUDA + TF32) for the verdict, and separately records a
sensitivity probe (full fp32 on the same device) so the sensitivity is documented rather than
discovered later.

Usage: python results/stage2_RQ-VAE/r45_gradient_integrity_audit/verify_sid_from_checkpoint.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
AUDIT = REPO / "results/stage2_RQ-VAE/r45_gradient_integrity_audit"
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K  # noqa: E402

ARMS = ["r4_08"] + [f"r5_{i:02d}" for i in range(1, 11)]
OUT = AUDIT / "sid_reproduction.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def configure(precision: str) -> None:
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = precision != "highest"
    torch.backends.cudnn.allow_tf32 = True


def export_sid(T, state_dict, embeddings, device: str) -> np.ndarray:
    model = T.RQVAE(T._tokenizer_config(), in_dim=embeddings.shape[1])
    model.load_state_dict(state_dict)
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    model = model.to(device)
    with torch.no_grad():
        raw = model.get_indices(torch.from_numpy(embeddings).to(device),
                                infer_use_sk=False).cpu().numpy().astype(np.int64)
    sizes = [int(size) for size in T.CODEBOOK_SIZE]
    if len(sizes) == 1:
        sizes *= T.CODEBOOK_NUM
    return T._extend_collisions(raw, sizes), raw


def verify(arm: str) -> dict:
    record: dict = {"arm": arm, "device": DEVICE, "verdict": "unverifiable", "detail": ""}
    try:
        T = K.load_arm(arm)
        arm_dir = REPO / f"results/stage2_RQ-VAE/{arm}"
        ckpt_path = arm_dir / "rqvae_best.pth"
        sid_path = arm_dir / "item_sids.json"
        if not ckpt_path.is_file():
            record["detail"] = f"missing checkpoint {ckpt_path}"
            return record
        if not sid_path.is_file():
            record["detail"] = f"missing item_sids.json {sid_path}"
            return record

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        record["checkpoint_epoch"] = payload.get("epoch")
        record["checkpoint_selection"] = payload.get("selection")
        record["checkpoint_collision"] = payload.get("collision")
        if payload.get("epoch") != 3000 or payload.get("selection") != "final_epoch":
            record["verdict"] = "FAILED"
            record["detail"] = (f"checkpoint metadata is epoch={payload.get('epoch')} "
                                f"selection={payload.get('selection')}, expected 3000/final_epoch")
            return record

        embeddings = np.asarray(T.maybe_apply_pca(
            T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED), dtype="float32")
        record["embedding_shape"] = list(embeddings.shape)
        record["embedding_sha_source"] = str(T.EMBEDDING_FILE)

        committed = json.loads(sid_path.read_text(encoding="utf-8"))
        n = len(committed)
        reference = np.asarray([committed[str(i)] for i in range(n)], dtype=np.int64)

        configure("high")                       # the trainer's production setting (TF32 matmuls)
        sid, raw = export_sid(T, payload["state_dict"], embeddings, DEVICE)
        record["production_config"] = {"precision": "high",
                                       "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                                       "device": DEVICE}
        record["shape_match"] = bool(sid.shape == reference.shape)
        if not record["shape_match"]:
            record["verdict"] = "FAILED"
            record["detail"] = f"shape {sid.shape} != committed {reference.shape}"
            return record
        record["exact_match"] = bool(np.array_equal(sid, reference))
        record["differing_cells"] = int((sid != reference).sum())
        record["differing_rows"] = int((sid != reference).any(axis=1).sum())
        record["total_cells"] = int(reference.size)

        # documented sensitivity: the same export under full fp32 precision
        configure("highest")
        sid_fp32, _ = export_sid(T, payload["state_dict"], embeddings, DEVICE)
        record["fp32_sensitivity_cells"] = int((sid_fp32 != reference).sum())
        record["fp32_sensitivity_rows"] = int((sid_fp32 != reference).any(axis=1).sum())
        configure("high")

        if record["exact_match"]:
            record["verdict"] = "VERIFIED"
            record["detail"] = (
                f"checkpoint state_dict re-exports the committed SID exactly under the "
                f"production configuration ({n} items x {sid.shape[1]} tokens, 0 differing "
                f"cells); under full fp32 {record['fp32_sensitivity_rows']}/{n} items flip, "
                "so the export is bit-reproducible only at the trainer's TF32 setting")
        else:
            record["verdict"] = "FAILED"
            record["detail"] = (f"{record['differing_cells']}/{reference.size} SID cells differ "
                                "under the production configuration")
        return record
    except Exception as exc:  # noqa: BLE001 - unverifiable, never 'passed'
        record["verdict"] = "unverifiable"
        record["detail"] = f"{type(exc).__name__}: {exc}"
        return record


def main() -> int:
    print(f"device={DEVICE} torch={torch.__version__}", flush=True)
    records = []
    for arm in ARMS:
        record = verify(arm)
        records.append(record)
        print(f"{arm}: {record['verdict']:12s} epoch={record.get('checkpoint_epoch')} "
              f"exact={record.get('exact_match')} cells={record.get('differing_cells')} "
              f"fp32_sensitive_rows={record.get('fp32_sensitivity_rows')}/{record.get('total_cells', 0) and len(json.loads((REPO / f'results/stage2_RQ-VAE/{arm}/item_sids.json').read_text()))}",
              flush=True)
    OUT.write_text(json.dumps({"device": DEVICE, "torch": torch.__version__,
                               "arms": records}, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for record in records:
        counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
    print(f"\nsummary: {counts}", flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0 if counts.get("FAILED") else 1


if __name__ == "__main__":
    sys.exit(main())
