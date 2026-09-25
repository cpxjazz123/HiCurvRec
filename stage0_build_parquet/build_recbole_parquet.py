"""Stage 0 preparation for the GeneRec recommendation pipeline.

Reads the raw Amazon-2023 Instruments dumps and emits the four Parquet
artifacts the downstream stages need:

1. ``items.parquet``               -- consumed by stage1 (sentence-T5
   item embeddings).  One row per item, with the four text fields
   (``title``, ``categories``, ``features``, ``description``) preserved
   as Python lists of strings exactly as RecBole3.0's
   ``build_metadata_text`` expects.

2. ``train.parquet``               -- consumed by stage3 (T5 sequence
   trainer).  Prefix-expanded user histories; the last two asins of
   each user are reserved for the valid/test holdouts (mirrors the
   HG-Rec TIGER convention).

3. ``valid.parquet``               -- consumed by stage3.  ``target =
   session[-2]``, ``history = session[:-2]``.

4. ``test.parquet``                -- consumed by stage3.  ``target =
   session[-1]``, ``history = session[:-1]``.

Replaces the original ``HG-Rec_amazon2023/export_recbole_tiger_data.py``
which delegated to RecBole3.0's ``compose_config`` + ``prepare`` chain.
The original pipeline pulled in a chain of incompatible dependencies on
this host (removed ``BeamSearchScorer`` in ``transformers`` >= 4.46,
etc.), so this script reads the raw JSON directly and rebuilds the
exact schema the existing parquet files use:

  * ``Instruments.inter.json`` is ``{user_id_str: [asin, asin, ...]}``
    one ordered asin list per user, kept verbatim (no dense remap).
  * ``Instruments.item.json`` is ``{item_id_str: {title, ...}}``.

Verified against the original parquet via reverse-engineering user=0:

    user=0 train targets = [12836, 12580, 12344]
                          = session[:-2] = first len-2 asins
    user=0 valid         = target=14247, history=[12836 12580 12344]
                          = session[-2], history=session[:-2]
    user=0 test          = target=6650, history=[12836 12580 12344 14247]
                          = session[-1], history=session[:-1]

where ``interactions["0"] = ['12836', '12580', '12344', '14247', '6650']``.

All paths are hard-coded (CLAUDE.md §1, no argparse / no env-var
override).  Only ``pandas`` / ``pyarrow`` are required, so the script
runs under any Python 3.8+ environment with the project's site-packages
on ``sys.path``.

Run from anywhere:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \\
        /home/wlia0047/ar57/wenyu/GeneRec/stage0_build_parquet/build_recbole_parquet.py
"""

import json
from pathlib import Path

import pandas as pd


# === Hard-coded knobs (CLAUDE.md §1, no argparse / no env-var override) ===
DATASET_DIR  = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Amazon_2023_Instruments"
)
INTER_JSON   = DATASET_DIR / "Instruments.inter.json"
ITEM_JSON    = DATASET_DIR / "Instruments.item.json"

OUTPUT_DIR   = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet"
)
ITEMS_FILE   = "items.parquet"
TRAIN_FILE   = "train.parquet"
VALID_FILE   = "valid.parquet"
TEST_FILE    = "test.parquet"

# Holdout policy: keep the last two interactions for valid + test,
# the rest go into train with prefix-based history building.  This
# matches the HG-Rec TIGER convention (see train_HG-Rec.py) and was
# empirically verified against the parquet files committed on 2026-09-22.
TRAIN_TARGETS_END_OFFSET = -2

# Metadata field order matches stage1's `build_metadata_text` (mirrors
# RecBole3.0 dataset/amazon2023/utils.py::build_metadata_text).  Restored
# here so stage1 does not need to re-derive the field order from a
# removed source.
METADATA_FIELDS = ("title", "categories", "features", "description")
# ===================================================================


def _user_to_int_id(interactions: dict) -> tuple[dict, int]:
    """Map user_id_str (a JSON object key) to a zero-based int.

    Uses Python 3.7+ dict insertion order (the original JSON-document
    user enumeration), so ``user=0`` corresponds to the first user
    object key in ``interactions``.  Lexicographic ordering diverges
    from that source order -- only insertion order matches the existing
    parquet files.
    """
    return (
        {str_user: int_idx for int_idx, str_user in enumerate(interactions.keys())},
        len(interactions),
    )


def _build_items_frame(raw_items: dict) -> pd.DataFrame:
    """Build the items frame.

    ``Instruments.item.json`` is ``{item_id_str: {title, ...}}``.  Each
    metadata field may be ``None``, a string, or a list of strings; we
    normalise the latter two as Python lists (keeping ``None`` as
    ``None`` so ``build_metadata_text`` short-circuits via
    ``pd.isna(value)``).  ``item_id`` becomes an ``int64`` for
    deterministic ordering, matching RecBole3.0's behaviour.
    """
    rows = []
    sorted_iids = sorted(int(iid) for iid in raw_items.keys())
    for iid in sorted_iids:
        meta = raw_items.get(str(iid)) or {}
        rows.append(
            {
                "item_id":     int(iid),
                **{field: meta.get(field) for field in METADATA_FIELDS},
            }
        )
    return pd.DataFrame(rows)


def _build_splits(
    interactions: dict, user_to_int: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the three split frames.

    For each user ``u`` with asin session ``[a0, a1, a2, ..., aN-1]``:

    * train rows: ``[(hist=[], tgt=a0),
                      (hist=[a0], tgt=a1),
                      ...
                      (hist=[a0..aN-3], tgt=aN-2-1)]``
      i.e. prefix expansion over the first ``N - 2`` asins (the last
      two are reserved for valid + test).
    * valid row: ``(hist=[a0..aN-3], seen_hist=[a0..aN-3], tgt=aN-2)``.
    * test  row: ``(hist=[a0..aN-2], seen_hist=[a0..aN-2], tgt=aN-1)``.

    Users with fewer than 3 interactions contribute zero train rows;
    with exactly 2 contribute valid only; with 1 contribute test only.
    """
    train_user, train_hist, train_seen, train_target = [], [], [], []
    valid_user, valid_hist, valid_seen, valid_target = [], [], [], []
    test_user,  test_hist,  test_seen,  test_target  = [], [], [], []

    for str_user, session in interactions.items():
        int_user = user_to_int[str_user]
        n = len(session)
        if n == 0:
            continue

        # The original HG-Rec export (see git log 052fe53 commit on the
        # HG-Rec_amazon2023 branch) casts every asin string to ``int64``
        # before serialising to parquet, so both ``history`` elements and
        # ``target`` end up as int64 inside ``list<element: int64>``.
        int_session = [int(asin) for asin in session]

        prefix_end = n + TRAIN_TARGETS_END_OFFSET  # = n - 2
        for i in range(prefix_end):
            history = int_session[:i]
            train_user.append(int_user)
            train_hist.append(history)
            train_seen.append(history)
            train_target.append(int_session[i])

        if n >= 2:
            hist_v = int_session[:-2]
            valid_user.append(int_user)
            valid_hist.append(hist_v)
            valid_seen.append(hist_v)
            valid_target.append(int_session[-2])

        if n >= 1:
            hist_t = int_session[:-1]
            test_user.append(int_user)
            test_hist.append(hist_t)
            test_seen.append(hist_t)
            test_target.append(int_session[-1])

    def _to_frame(u, h, s, t):
        return pd.DataFrame(
            {
                "user":         pd.Series(u, dtype="int64"),
                "history":      pd.Series(h, dtype="object"),
                "seen_history": pd.Series(s, dtype="object"),
                "target":       pd.Series(t, dtype="int64"),
            }
        )

    return (
        _to_frame(train_user, train_hist, train_seen, train_target),
        _to_frame(valid_user, valid_hist, valid_seen, valid_target),
        _to_frame(test_user,  test_hist,  test_seen,  test_target),
    )


def main() -> int:
    if not INTER_JSON.is_file():
        raise FileNotFoundError(f"Raw interaction dump missing: {INTER_JSON}")
    if not ITEM_JSON.is_file():
        raise FileNotFoundError(f"Raw item-metadata dump missing: {ITEM_JSON}")

    with INTER_JSON.open("r", encoding="utf-8") as handle:
        interactions = json.load(handle)
    if not isinstance(interactions, dict) or not interactions:
        raise ValueError(f"Instruments.inter.json must be a non-empty object: {INTER_JSON}")

    with ITEM_JSON.open("r", encoding="utf-8") as handle:
        raw_items = json.load(handle)
    if not isinstance(raw_items, dict) or not raw_items:
        raise ValueError(f"Instruments.item.json must be a non-empty object: {ITEM_JSON}")

    user_to_int, n_users = _user_to_int_id(interactions)
    items_frame              = _build_items_frame(raw_items)
    train_frame, valid_frame, test_frame = _build_splits(interactions, user_to_int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items_frame.to_parquet(OUTPUT_DIR / ITEMS_FILE, index=False)
    train_frame.to_parquet(OUTPUT_DIR / TRAIN_FILE, index=False)
    valid_frame.to_parquet(OUTPUT_DIR / VALID_FILE, index=False)
    test_frame.to_parquet(OUTPUT_DIR / TEST_FILE,  index=False)

    n_items = len(items_frame)
    print(
        f"Built stage0 parquet from {INTER_JSON.name} + {ITEM_JSON.name}: "
        f"users={n_users} items={n_items} "
        f"train={len(train_frame)} valid={len(valid_frame)} test={len(test_frame)}"
    )
    print(f"Output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
