"""Generate per-item text embeddings as RQ-VAE input.

Self-contained: this script does not import any other file from the repository
and exposes no CLI arguments -- all paths and encoder choices are hard-coded
in the CONFIG block below.

This script mirrors the RecBole3.0 RQ-VAE Stage 0 embedding pipeline verbatim:

  - text construction = recbole3/dataset/amazon2023/utils.py::build_metadata_text
    (title, categories, features, description -> " ".join(feature_to_sentence(.)))
    plus the full clean_text() pre-processing (html unescape, strip <tags>,
    collapse whitespace, drop non-ASCII).
  - encoder           = sentence-transformers/sentence-t5-base, sem_emb_dim=768
    (recbole3/model/rqvae/data.py::_generate_semantic_embeddings, default config
     from recbole3/model/rqvae/config.py).
  - batch size        = 32 (recbole3/model/rqvae/config.py::sent_emb_batch_size)
  - device            = cuda
  - output            = (N, 768) float32 npy at <stage>/output/sentence_t5.npy
    (recbole3/model/rqvae/config.py::sem_emb_file = "sentence_t5.npy")

For 2023 Instruments the source JSON only carries title + description; the
other two fields are simply absent and skipped by build_metadata_text.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIG (hard-coded; mirrors RecBole3.0 defaults)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# recbole3/model/rqvae/config.py::RQVAEConfig defaults
ENCODER_ID = "sentence-transformers/sentence-t5-base"
SEM_EMB_DIM = 768
SENT_EMB_BATCH_SIZE = 32
DEVICE = "cuda"

# recbole3/dataset/amazon2023/utils.py::build_metadata_text field order
METADATA_FIELDS = ("title", "categories", "features", "description")

INPUT_PATH = REPO_ROOT / "dataset" / "Amazon_2023_Instruments" / "Instruments.item.json"

OUTPUT_DIR = SCRIPT_DIR / "output"
NPY_NAME = "sentence_t5.npy"      # RecBole3.0 default (sem_emb_file)
# ---------------------------------------------------------------------------


# ---- text utilities (verbatim port of RecBole3.0 utils) --------------------

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
_MULTISPACE_RE = re.compile(r" +")
_NEWLINE_TAB_RE = re.compile(r"[\n\t]")
_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(raw_text):
    text = stringify_feature(raw_text)
    text = html.unescape(text).strip()
    text = _TAG_RE.sub("", text)
    text = _NEWLINE_TAB_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _NON_ASCII_RE.sub(" ", text)
    return text.strip()


def stringify_feature(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, list):
        cleaned_values = [clean_text(item) for item in value]
        return ", ".join(item for item in cleaned_values if item)
    return str(value)


def feature_to_sentence(value):
    cleaned_value = clean_text(value)
    if not cleaned_value:
        return ""
    return f"{cleaned_value}."


def build_metadata_text(row):
    sentences = [feature_to_sentence(row.get(field)) for field in METADATA_FIELDS]
    return " ".join(sentence for sentence in sentences if sentence).strip()


# ---- item loading + encoding ----------------------------------------------


def _read_items(path: Path) -> tuple[list[int], list[str]]:
    """Return (sorted item_ids, metadata_text per item).

    Mirrors recbole3/model/rqvae/data.py::_generate_semantic_embeddings:
      - load JSON dict {item_id_str: {...}}
      - sort by int(item_id)
      - texts[i] = build_metadata_text(items[str(i)])
    """
    with path.open("r", encoding="utf-8") as handle:
        items = json.load(handle)

    item_ids = sorted(int(iid) for iid in items.keys())
    texts = [
        build_metadata_text(items[str(iid)] or {}) for iid in item_ids
    ]
    return item_ids, texts


def _generate_semantic_embeddings(texts: list[str]) -> np.ndarray:
    """Mirror recbole3/model/rqvae/data.py::_generate_semantic_embeddings."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "[stage1_GeneEmbedding] sentence-transformers is required. "
            "Install with: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(ENCODER_ID)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=SENT_EMB_BATCH_SIZE,
        show_progress_bar=True,
        device=DEVICE,
    )
    return np.asarray(embeddings, dtype=np.float32)


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"[stage1_GeneEmbedding] input file not found: {INPUT_PATH}"
        )

    item_ids, texts = _read_items(INPUT_PATH)
    n_items = len(item_ids)
    print(
        f"[stage1_GeneEmbedding] {n_items} items loaded from "
        f"{INPUT_PATH.name}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = OUTPUT_DIR / NPY_NAME
    meta_path = OUTPUT_DIR / "emb_meta.json"

    embeddings = _generate_semantic_embeddings(texts)

    if embeddings.shape != (n_items, SEM_EMB_DIM):
        raise SystemExit(
            f"[stage1_GeneEmbedding] embedding shape {embeddings.shape} != "
            f"expected ({n_items}, {SEM_EMB_DIM})"
        )

    np.save(npy_path, embeddings)
    meta = {
        "encoder": ENCODER_ID,
        "sem_emb_dim": SEM_EMB_DIM,
        "batch_size": SENT_EMB_BATCH_SIZE,
        "device": DEVICE,
        "metadata_fields": list(METADATA_FIELDS),
        "input_path": str(INPUT_PATH),
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "item_id_min": int(min(item_ids)),
        "item_id_max": int(max(item_ids)),
        "output_npy": str(npy_path),
        "source": "RecBole3.0/src/recbole3/{model/rqvae,dataset/amazon2023}",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[stage1_GeneEmbedding] wrote {embeddings.shape} -> {npy_path}; "
        f"meta -> {meta_path}"
    )


if __name__ == "__main__":
    main()
