"""Audit the frozen TIGER SID artefacts (issue #246 §7, no training).

Three questions:
  1. what exactly is in each legacy file (schema, item mapping, order)?
  2. does a read-only re-export from the frozen checkpoint reproduce one of them?
  3. which historical Stage3 recall can honestly be bound to a SID hash?

Everything is written to a new directory; no historical artefact is modified.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.graph_infra import metrics  # noqa: E402

ARM_DIR = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
RESULT_DIR = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
OUT_DIR = REPO / "results" / "research" / "infra246" / "tiger_reexport"
JSON_SID = RESULT_DIR / "item_sids.json"
NPY_SID = RESULT_DIR / "sids_for_hgrec.npy"
CHECKPOINT = RESULT_DIR / "rqvae_best.pth"
EMBEDDINGS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_matrix(matrix: np.ndarray) -> dict:
    """Schema of an (n, k) integer matrix: per-column ranges and uniqueness."""
    columns = []
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        columns.append({"column": index, "min": int(column.min()), "max": int(column.max()),
                        "distinct": int(np.unique(column).size),
                        "is_sequential_code": bool(column.min() >= 0
                                                  and column.max() < 256 * 64)})
    return {"shape": list(matrix.shape), "dtype": str(matrix.dtype), "columns": columns,
            "distinct_rows": int(np.unique(matrix, axis=0).shape[0]),
            "distinct_prefix3": int(np.unique(matrix[:, :3], axis=0).shape[0])}


def structure_of(matrix: np.ndarray) -> dict:
    """Collision and duplicate rates under both calibers."""
    report = metrics.sid_structure(matrix.tolist())
    return {"collision_prefix3": report["collision_prefix3"],
            "duplicate_prefix3": report["duplicate_prefix3"],
            "collision_full": report["collision_full"],
            "duplicate_full": report["duplicate_full"],
            "distinct_prefix3": report["distinct_prefix3"],
            "distinct_full": report["distinct_full"],
            "items": report["total"]}


def read_json_table(path: Path) -> tuple[np.ndarray, list[int]]:
    table = json.loads(path.read_text(encoding="utf-8"))
    keys = sorted(int(key) for key in table)
    rows = [list(table[str(key)]) for key in keys]
    return np.array(rows, dtype=np.int64), keys


def locate_npy_origin(matrix: np.ndarray) -> dict:
    """Which other arm's export does the NPY correspond to?"""
    candidates = {}
    target = {tuple(row[:3]) for row in matrix.tolist()}
    for path in sorted((REPO / "results" / "stage2_RQ-VAE").glob("*/item_sids.json")):
        try:
            other = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if len(other) != matrix.shape[0]:
            continue
        triples = {tuple(int(x) for x in row[:3]) for row in other.values()}
        overlap = len(target & triples) / max(1, len(target))
        if overlap > 0.5:
            candidates[path.parent.name] = overlap
    return {"best_match": max(candidates, key=candidates.get) if candidates else None,
            "matches": dict(sorted(candidates.items(), key=lambda item: -item[1])[:5])}


def reexport() -> dict:
    """Read-only re-export through the arm's own export code."""
    sys.path.insert(0, str(ARM_DIR))
    import train_rqvae  # noqa: E402  (the frozen arm's own export entry)
    from model import RQVAE  # noqa: E402

    config = train_rqvae._tokenizer_config()
    model = RQVAE(config, in_dim=768, distance=train_rqvae.DISTANCE)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    embeddings = np.load(EMBEDDINGS).astype(np.float32)
    tensor = torch.from_numpy(embeddings)
    with torch.no_grad():
        encoded = model.encoder(tensor)
        reconstructed, _, _, tokens = model.rq.forward(encoded) if False else (None, None, None, None)
    with torch.no_grad():
        raw_tokens = model.get_indices(tensor, infer_use_sk=False).numpy().astype(np.int64)
        reconstruction, _, _, _ = model(tensor)
    sid = train_rqvae._extend_collisions(raw_tokens, list(config.codebook_size))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "sids_for_hgrec.npy", sid)
    payload = {str(item_id): [int(value) for value in row] for item_id, row in enumerate(sid)}
    (OUT_DIR / "item_sids.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    error = reconstruction.numpy().astype(np.float64) - embeddings.astype(np.float64)
    return {
        "raw_tokens_shape": list(raw_tokens.shape),
        "sid_shape": list(sid.shape),
        "distinct_rows": int(np.unique(sid, axis=0).shape[0]),
        "checkpoint": {"epoch": checkpoint.get("epoch"), "collision": checkpoint.get("collision"),
                       "selection": checkpoint.get("selection"), "sha256": sha256(CHECKPOINT)},
        "reconstruction_mse": float((error ** 2).mean()),
        "reconstruction_mae": float(np.abs(error).mean()),
        "reconstruction_unit": "squared error on the 768-dim input embeddings",
        "item_sids_sha256": sha256(OUT_DIR / "item_sids.json"),
        "sids_for_hgrec_sha256": sha256(OUT_DIR / "sids_for_hgrec.npy"),
    }


def compare(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        return {"same_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    differing = np.any(left != right, axis=1)
    return {"same_shape": True, "rows_differing": int(differing.sum()),
            "first_differing": int(np.flatnonzero(differing)[0]) if differing.any() else None,
            "identical": bool(not differing.any())}


def stage3_binding() -> dict:
    """Which SID did the historical Stage3 run actually read?"""
    directory = REPO / "results" / "stage3_T5Train" / "TIGER_RQ-VAE"
    references = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in (".log", ".json", ".txt", ".jsonl"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            for line in text.splitlines():
                if "item_sids" in line or "sids_for_hgrec" in line:
                    references.append({"file": str(path.relative_to(REPO)), "line": line.strip()[:200]})
    return {"references": references[:20], "count": len(references)}


def main() -> int:
    report = {"arm": "TIGER_RQ-VAE", "read_only": True}
    json_matrix, json_keys = read_json_table(JSON_SID)
    npy_matrix = np.load(NPY_SID)
    report["legacy"] = {
        "item_sids.json": {"sha256": sha256(JSON_SID), "schema": describe_matrix(json_matrix),
                           "structure": structure_of(json_matrix),
                           "keys_contiguous": json_keys == list(range(len(json_keys))),
                           "key_min": min(json_keys), "key_max": max(json_keys)},
        "sids_for_hgrec.npy": {"sha256": sha256(NPY_SID), "schema": describe_matrix(npy_matrix),
                               "structure": structure_of(npy_matrix)},
        "json_vs_npy": compare(json_matrix, npy_matrix),
        "npy_origin": locate_npy_origin(npy_matrix),
    }
    report["reexport"] = reexport()
    fresh = np.load(OUT_DIR / "sids_for_hgrec.npy")
    report["legacy"]["reexport_vs_item_sids_json"] = compare(json_matrix, fresh)
    report["legacy"]["reexport_vs_sids_for_hgrec_npy"] = compare(npy_matrix, fresh)
    report["legacy"]["reexport_structure"] = structure_of(fresh)
    report["stage3_binding"] = stage3_binding()
    report["recall_binding"] = {
        "test_R@10": "NA: the historical Stage3 run did not record a SID hash, so the number "
                     "cannot be attributed to a specific SID table; see stage3_binding.references",
    }
    out = REPO / "results" / "research" / "infra246" / "tiger_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"legacy": {name: report["legacy"][name]["structure"]
                                 for name in ("item_sids.json", "sids_for_hgrec.npy")},
                      "json_vs_npy": report["legacy"]["json_vs_npy"],
                      "reexport_vs_json": report["legacy"]["reexport_vs_item_sids_json"],
                      "reexport_vs_npy": report["legacy"]["reexport_vs_sids_for_hgrec_npy"],
                      "reconstruction_mse": report["reexport"]["reconstruction_mse"],
                      "npy_origin": report["legacy"]["npy_origin"]["best_match"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
