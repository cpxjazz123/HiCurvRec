#!/usr/bin/env python3
"""
Task #73 — Convert Musical_Instruments.inter (RecBole format) to LLM-RecSys-ID format.

Input:  data/recbole/Musical_Instruments.inter
        Format: user_id:token \t item_id:token \t rating:float \t timestamp:float

Output: external/LLM-RecSys-ID/data/instruments/remapped_sequential_data.txt
        Format: user_id item_1 item_2 ... item_N (space-separated, no trailing)

Mapping:
- RecBole user_id is already 0-indexed contiguous → use directly
- RecBole item_id is 0-indexed contiguous → use directly
- Group by user_id, sort by timestamp, write sequence (all items, leave-one-out handled by P5 framework)
"""
import os
import sys
from collections import defaultdict

INPUT_FILE = "/home/wlia0047/ar57/wenyu/GeneRec/data/recbole/Musical_Instruments/Musical_Instruments.inter"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/instruments"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "remapped_sequential_data.txt")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input not found: {INPUT_FILE}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[1/3] Reading {INPUT_FILE}...")
    user_sequences = defaultdict(list)
    with open(INPUT_FILE) as f:
        header = f.readline().strip().split("\t")
        uid_idx = header.index("user_id:token")
        iid_idx = header.index("item_id:token")
        ts_idx = header.index("timestamp:float")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            uid = int(parts[uid_idx])
            iid = int(parts[iid_idx])
            ts = float(parts[ts_idx])
            user_sequences[uid].append((ts, iid))

    print(f"[2/3] Sorting sequences by timestamp...")
    n_users = len(user_sequences)
    n_interactions = sum(len(seq) for seq in user_sequences.values())
    print(f"  Users: {n_users}")
    print(f"  Interactions: {n_interactions}")
    print(f"  Avg seq len: {n_interactions / max(1, n_users):.1f}")

    # Filter users with < 5 interactions (5-core)
    print("[3/3] Writing output (5-core filter + sort by timestamp)...")
    kept_users = 0
    total_seqs_written = 0
    with open(OUTPUT_FILE, "w") as f:
        for uid in sorted(user_sequences.keys()):
            seq = sorted(user_sequences[uid], key=lambda x: x[0])
            items = [str(iid) for _, iid in seq]
            if len(items) < 5:
                continue
            f.write(f"{uid} " + " ".join(items) + "\n")
            kept_users += 1
            total_seqs_written += len(items)

    print(f"✅ Done. Wrote {kept_users} users / {total_seqs_written} interactions to {OUTPUT_FILE}")
    print(f"  Output avg seq len: {total_seqs_written / max(1, kept_users):.1f}")


if __name__ == "__main__":
    main()
