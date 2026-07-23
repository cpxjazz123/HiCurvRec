#!/usr/bin/env python3
"""
Task #72 Phase 4: ETEGRec Data Preparation for Musical_Instruments

Converts DECOR processed data → ETEGRec JSONL format.

ETEGRec expects:
  ./dataset/{dataset}/{dataset}.{train,valid,test}.jsonl
    Each line: {"inter_history": [token, ...], "target_id": token}
  ./dataset/{dataset}/{dataset}.emb_map.json
    Dict: token -> integer ID (1-indexed)
  ./dataset/{dataset}/{dataset}_emb_{dim}.npy
    Embeddings shape (n_items+1, dim) — index 0 unused

Split strategy (leave-one-out):
  For user history [i1, i2, ..., i_{n-2}, i_{n-1}, i_n]:
    Train: ALL prefix-suffix pairs where target != i_n and != i_{n-1}
           e.g. [i1]->i2, [i1,i2]->i3, ..., [i1,...,i_{n-3}]->i_{n-2}
           AND [i1,...,i_{n-2}]->i_{n-1} (valid pair also seen in training)
    Valid: [i1,...,i_{n-2}] -> i_{n-1}
    Test:  [i1,...,i_{n-1}] -> i_n
"""

import json
import os
import numpy as np

# === Paths ===
PROCESSED_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/DECOR/cache/AmazonReviews2023/Musical_Instruments/processed"
ETEGREC_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/ETEGRec"
OUTPUT_DIR = os.path.join(ETEGREC_DIR, "dataset/Musical_Instruments")

ALL_ITEM_SEQS = os.path.join(PROCESSED_DIR, "all_item_seqs.json")
ID_MAPPING = os.path.join(PROCESSED_DIR, "id_mapping.json")
EMB_NPY = os.path.join(PROCESSED_DIR, "sentence-t5-base.sent_emb.npy")

TRAIN_JSONL = os.path.join(OUTPUT_DIR, "Musical_Instruments.train.jsonl")
VALID_JSONL = os.path.join(OUTPUT_DIR, "Musical_Instruments.valid.jsonl")
TEST_JSONL = os.path.join(OUTPUT_DIR, "Musical_Instruments.test.jsonl")
EMB_MAP_JSON = os.path.join(OUTPUT_DIR, "Musical_Instruments.emb_map.json")
EMB_NPY_OUT = os.path.join(OUTPUT_DIR, "Musical_Instruments_emb_128.npy")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("ETEGRec Data Preparation for Musical_Instruments")
print("=" * 60)

# === Step 1: Load data ===
print("\n[1] Loading all_item_seqs.json...")
with open(ALL_ITEM_SEQS, "r") as f:
    all_seqs = json.load(f)
print(f"  Users: {len(all_seqs)}")

print("[2] Loading id_mapping.json...")
with open(ID_MAPPING, "r") as f:
    mapping = json.load(f)
item2id_asin = mapping.get("item2id", {})
print(f"  Items in mapping: {len(item2id_asin)}")

# === Step 2: Build sorted item set from sequences ===
print("[3] Building item set from sequences...")
all_items_set = set()
user_histories = []
for user_id, history in all_seqs.items():
    all_items_set.update(history)
    user_histories.append(history)
item_tokens = sorted(all_items_set)
print(f"  Unique items: {len(item_tokens)}")
print(f"  In mapping: {sum(1 for t in item_tokens if t in item2id_asin)}")

# ETEGRec: 1-indexed item IDs
item_to_etegrec_id = {asin: i + 1 for i, asin in enumerate(item_tokens)}

# === Step 3: Split histories (leave-one-out) ===
print("[4] Splitting histories (leave-one-out)...")
train_data = []
valid_data = []
test_data = []
skipped_short = 0

for history in user_histories:
    n = len(history)
    if n < 2:
        skipped_short += 1
        continue

    # Test: last item from prefix of all-but-last
    test_data.append({
        "inter_history": history[:-1],
        "target_id": history[-1]
    })

    # Valid: second-last item from prefix of all-but-last-two
    if n >= 3:
        valid_data.append({
            "inter_history": history[:-2],
            "target_id": history[-2]
        })
    else:
        # n == 2: valid and test are the same
        valid_data.append({
            "inter_history": history[:-1],
            "target_id": history[-1]
        })

    # Train: ALL prefix-suffix pairs
    # For sequence of length n, pairs are:
    #   [0]->1, [0,1]->2, ..., [0,...,n-3]->n-2, [0,...,n-2]->n-1  (not n-1->n, that's test)
    for i in range(1, n):
        # Don't include the very last pair as training (that's test)
        if i == n - 1 and n > 2:
            # But DO include the second-last pair (that's also validation) for training
            # i == n-1 means: prefix = [0,...,n-2], target = n-1
            pass  # include it
        elif i == n - 1:
            # n == 2: history[:-1] = [0], target = 1 (this is both valid and test... skip for train)
            continue

        train_data.append({
            "inter_history": history[:i],
            "target_id": history[i]
        })

print(f"  Skipped (<2 items): {skipped_short}")
print(f"  Train samples: {len(train_data)}")
print(f"  Valid samples: {len(valid_data)}")
print(f"  Test samples:  {len(test_data)}")

# === Step 4: Write output files ===
print("\n[5] Writing output files...")

def write_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

write_jsonl(train_data, TRAIN_JSONL)
write_jsonl(valid_data, VALID_JSONL)
write_jsonl(test_data, TEST_JSONL)
print(f"  JSONL files written to {OUTPUT_DIR}")

with open(EMB_MAP_JSON, "w") as f:
    json.dump(item_to_etegrec_id, f, indent=2)
print(f"  emb_map: {len(item_to_etegrec_id)} entries -> {EMB_MAP_JSON}")

# === Step 5: Reindex embeddings ===
print(f"\n[6] Processing embeddings...")
embeddings = np.load(EMB_NPY)
print(f"  Original: {embeddings.shape}")

n_items = len(item_to_etegrec_id)
emb_dim = embeddings.shape[1]
emb_reindexed = np.zeros((n_items + 1, emb_dim), dtype=embeddings.dtype)
# ETEGRec uses 1-indexed IDs, so emb[0] = zero padding

missing = 0
out_of_range = 0
for asin, etegrec_id in item_to_etegrec_id.items():
    decor_idx = item2id_asin.get(asin)
    if decor_idx is not None:
        if decor_idx < len(embeddings):
            emb_reindexed[etegrec_id] = embeddings[decor_idx]
        else:
            out_of_range += 1
    else:
        missing += 1

print(f"  Missing from mapping: {missing}/{n_items}")
print(f"  Embedding index OOB: {out_of_range}/{n_items}")
print(f"  Output: {emb_reindexed.shape}")
np.save(EMB_NPY_OUT, emb_reindexed)
print(f"  Saved -> {EMB_NPY_OUT}")

# === Summary ===
print("\n" + "=" * 60)
print("SUMMARY")
print(f"  Items:      {n_items}")
print(f"  Embed dim:  {emb_dim}")
print(f"  Train:      {len(train_data)}")
print(f"  Valid:      {len(valid_data)}")
print(f"  Test:       {len(test_data)}")
print(f"  Output dir: {OUTPUT_DIR}")
print("=" * 60)
