"""Generate per-item text embeddings as RQ-VAE input.

This is **stage1** of the GeneRec pipeline.  It consumes the
``items.parquet`` artifact emitted by stage0 (see
``stage0_build_parquet/build_recbole_parquet.py``) and writes a
``sentence_t5.npy`` plus a small ``emb_meta.json`` metadata file that
the stage2 RQ-VAE trainer reads.

Schema contract (stage0 -> stage1):

  * ``items.parquet`` is a Parquet frame with one row per item.
  * Columns: ``item_id: int64`` plus one column per metadata field
    (currently ``title``, ``categories``, ``features``, ``description``).
    Each field column may hold a Python ``str`` or ``list[str]``;
    ``None`` is allowed and is short-circuited by ``build_metadata_text``.
  * Sort order = ``int(item_id)`` ascending.

This script mirrors the RecBole3.0 RQ-VAE text pipeline:

  - text construction = recbole3/dataset/amazon2023/utils.py::build_metadata_text
    (title, categories, features, description -> " ".join(feature_to_sentence(.)))
    plus the full clean_text() pre-processing (html unescape, strip <tags>,
    collapse whitespace, drop non-ASCII).
  - encoder           = sentence-transformers/sentence-t5-base, sem_emb_dim=768
    (recbole3/model/rqvae/data.py::_generate_semantic_embeddings, default config
     from recbole3/model/rqvae/config.py).
  - batch size        = 32
  - device            = cuda
  - output            = (N, 768) float32 npy at <stage>/output/sentence_t5.npy

Self-contained: this script does not import any other file from the
repository and exposes no CLI arguments -- all paths and encoder choices
are hard-coded in the CONFIG block below.

Run from anywhere:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \\
        /home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/generate_item_embedding.py
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

# Stage0 output we depend on -- do not change without re-running stage0.
INPUT_PATH = REPO_ROOT / "results" / "stage0_build_parquet" / "items.parquet"

OUTPUT_DIR = SCRIPT_DIR / "output"
NPY_NAME = "sentence_t5.npy"      # RecBole3.0 default (sem_emb_file)
IDS_JSON_NAME = "item_ids.json"   # sidecar listing row order of sentence_t5.npy
                                   # (consumed by stage2/curvature_RQ-VAE)
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

    Loads the stage0-emitted ``items.parquet`` and applies the same
    metadata-text pipeline that RecBole3.0's RQ-VAE
    ``_generate_semantic_embeddings`` step used.
    """
    df = pd.read_parquet(path)
    required = {"item_id", *METADATA_FIELDS}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"stage0 items.parquet is missing required columns: {sorted(missing)}"
        )

    df = df.sort_values("item_id", kind="stable").reset_index(drop=True)
    item_ids = df["item_id"].astype("int64").tolist()
    texts    = [
        build_metadata_text(df.iloc[i].to_dict()) for i in range(len(df))
    ]
    return item_ids, texts


def _generate_semantic_embeddings(texts: list[str]) -> np.ndarray:
    """Encode ``texts`` with sentence-T5-base via plain ``transformers``.

    Equivalent to ``sentence_transformers.SentenceTransformer(ENCODER_ID).encode``
    on a sentence-T5 model: mean-pool the encoder's last_hidden_state with
    the attention mask, returning one (D,) vector per text.  This avoids
    the optional ``sentence-transformers`` dependency (which transitively
    pulls TensorFlow / Flax at import time and breaks on minimal envs).
    """
    try:
        import torch
        from transformers import T5EncoderModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "[stage1_GeneEmbedding] transformers + torch are required. "
            "Install with: pip install transformers torch"
        ) from exc

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ENCODER_ID, legacy=True)
    # ``sentence-transformers/sentence-t5-base`` ships only the encoder
    # side of T5, so use ``T5EncoderModel`` (decoder-free).  ``AutoModel``
    # would materialise a fresh decoder whose weights were never
    # pretrained, and ``model(**enc)`` would then error demanding
    # ``decoder_input_ids``.
    model = T5EncoderModel.from_pretrained(ENCODER_ID).to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), SENT_EMB_BATCH_SIZE):
            batch = texts[start:start + SENT_EMB_BATCH_SIZE]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            out = model(**enc).last_hidden_state            # (B, T, D)
            mask = enc["attention_mask"].unsqueeze(-1).type_as(out)
            summed = (out * mask).sum(dim=1)                # (B, D)
            counts = mask.sum(dim=1).clamp(min=1)           # (B, 1)
            mean_pooled = summed / counts                   # (B, D)
            embeddings.append(mean_pooled.cpu().float().numpy())

    return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"[stage1_GeneEmbedding] stage0 input file not found: {INPUT_PATH}. "
            f"Run stage0_build_parquet/build_recbole_parquet.py first."
        )

    item_ids, texts = _read_items(INPUT_PATH)
    n_items = len(item_ids)
    print(
        f"[stage1_GeneEmbedding] {n_items} items loaded from "
        f"{INPUT_PATH.relative_to(REPO_ROOT)}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = OUTPUT_DIR / NPY_NAME
    meta_path = OUTPUT_DIR / "emb_meta.json"
    ids_json_path = OUTPUT_DIR / IDS_JSON_NAME   # for stage2 RQ-VAE order pin

    embeddings = _generate_semantic_embeddings(texts)

    if embeddings.shape != (n_items, SEM_EMB_DIM):
        raise SystemExit(
            f"[stage1_GeneEmbedding] embedding shape {embeddings.shape} != "
            f"expected ({n_items}, {SEM_EMB_DIM})"
        )

    np.save(npy_path, embeddings)
    # The RQ-VAE trainer (stage2/curvature_RQ-VAE) aligns its item index
    # ordering with a sidecar item_ids.json so the model can pin row i to
    # the correct item id; mirror the row order of sentence_t5.npy.
    ids_json_path.write_text(json.dumps([int(i) for i in item_ids], indent=0))
    meta = {
        "encoder": ENCODER_ID,
        "sem_emb_dim": SEM_EMB_DIM,
        "batch_size": SENT_EMB_BATCH_SIZE,
        "device": DEVICE,
        "metadata_fields": list(METADATA_FIELDS),
        "input_path": str(INPUT_PATH),
        "ids_json": str(ids_json_path),
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "item_id_min": int(min(item_ids)),
        "item_id_max": int(max(item_ids)),
        "output_npy": str(npy_path),
        "source": "stage0 results/stage0_build_parquet/items.parquet; "
                  "RecBole3.0/{model/rqvae,dataset/amazon2023} field order preserved",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[stage1_GeneEmbedding] wrote {embeddings.shape} -> {npy_path}; "
        f"ids -> {ids_json_path}; meta -> {meta_path}"
    )


if __name__ == "__main__":
    main()
