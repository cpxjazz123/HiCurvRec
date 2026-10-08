"""Run frozen D14-D17 geometry diagnostics and merge them into sid_diag/summary.json."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
SOURCE_DIR = ROOT / "stage2_RQ-VAE/curvature_RQ-VAE"
OUTPUT_DIR = ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE/sid_diag"
AUDIT_SCRIPT = SOURCE_DIR / "scripts/sid_diag.py"
DIAGNOSTIC_SUMMARY = OUTPUT_DIR / "summary.json"

sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(SOURCE_DIR / "scripts"))
import curvature_config as experiment  # noqa: E402
import train_rqvae as trainer  # noqa: E402
import sid_diag as audit  # noqa: E402
from model import RQVAE  # noqa: E402


def main() -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    checkpoint_path = Path(experiment.RQVAE_CKPT_PATH)
    embedding_path = Path(experiment.EMBEDDING_FILE)
    train_path = Path(experiment.TRAIN_FILE)
    raw_sid_path = Path(experiment.RAW_SIDS_NPY)
    hgrec_sid_path = Path(experiment.SIDS_NPY)
    pair_path = OUTPUT_DIR / "behavior_pair_sample.csv"
    for path in (
        checkpoint_path, embedding_path, train_path, raw_sid_path,
        hgrec_sid_path, pair_path, DIAGNOSTIC_SUMMARY,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    embeddings_np = np.asarray(np.load(embedding_path), dtype=np.float32)
    raw_tokens = np.asarray(np.load(raw_sid_path), dtype=np.int64)
    hgrec_tokens = np.asarray(np.load(hgrec_sid_path), dtype=np.int64)[:, :3]
    if raw_tokens.shape != (len(embeddings_np), 3) or not np.array_equal(raw_tokens, hgrec_tokens):
        raise RuntimeError("Stage2 SID exports do not match the frozen item embedding order")
    summary = json.loads(DIAGNOSTIC_SUMMARY.read_text(encoding="utf-8"))
    if int(summary["metadata"]["checkpoint"]["global_step"]) != int(checkpoint["global_step"]):
        raise RuntimeError("Frozen diagnostic summary and checkpoint refer to different Stage2 steps")
    for path in (checkpoint_path, embedding_path, train_path, raw_sid_path, hgrec_sid_path):
        relative = audit.relative_to_root(path)
        recorded_hash = summary["metadata"]["input_sha256"].get(relative)
        if recorded_hash is not None and audit.sha256(path) != recorded_hash:
            raise RuntimeError(f"Frozen diagnostic input changed since SID-Diag-01: {relative}")

    train_frame = pd.read_parquet(train_path, columns=["seen_history", "target"])
    source_ids, successor_ids = trainer._transition_pairs(train_frame)
    embedding_tensor = torch.from_numpy(embeddings_np)
    behaviour = trainer._behaviour_context(embedding_tensor, source_ids, successor_ids)
    context_channel = trainer._curvature_context_channel(embedding_tensor, behaviour)
    encoder_inputs = torch.cat((embedding_tensor, context_channel), dim=1)

    config = trainer._tokenizer_config()
    model = RQVAE(
        config,
        in_dim=embeddings_np.shape[1],
        context_dim=embeddings_np.shape[1],
    ).cpu().eval()
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict Stage2 checkpoint load failed: {incompatible}")

    pair_rows = pd.read_csv(pair_path).to_dict(orient="records")
    diagnostics = audit.d14_d17_diagnostics(model, encoder_inputs, raw_tokens, pair_rows)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    audit_path = Path(__file__).resolve()
    audit_hash = audit.sha256(audit_path)
    script_hash = audit.sha256(AUDIT_SCRIPT)
    summary.setdefault("metadata", {}).setdefault("source_files_sha256", {})[
        audit.relative_to_root(AUDIT_SCRIPT)
    ] = script_hash
    summary["metadata"]["source_files_sha256"][audit.relative_to_root(audit_path)] = audit_hash
    summary["D14-D17"] = diagnostics
    summary["metadata"]["D14-D17"] = {
        "source_commit_before_diagnostics": source_commit,
        "diagnostic_script": audit.relative_to_root(audit_path),
        "diagnostic_script_sha256": audit_hash,
        "behavior_pair_sample_sha256": audit.sha256(pair_path),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "stage3_launched": False,
    }
    audit.write_json(DIAGNOSTIC_SUMMARY, summary)
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
