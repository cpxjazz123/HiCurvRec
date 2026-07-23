#!/usr/bin/env python3
"""
Task #72 Fix: 5-core filter for Musical_Instruments dataset
DECOR paper uses 5-core filter (each user/item ≥5 interactions).
We downloaded Amazon 2023 raw without filter → 2x data size → baseline R@10 inflated.

This script iteratively filters until all users and items have ≥5 interactions.
"""

import os
import sys
from collections import defaultdict

INPUT = "/home/wlia0047/ar57/wenyu/GeneRec/data/recbole/Musical_Instruments/Musical_Instruments.inter"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/data/recbole/Musical_Instruments_5core"
OUTPUT = os.path.join(OUTPUT_DIR, "Musical_Instruments.inter")
MIN_INTER = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read all interactions
interactions = []
with open(INPUT, 'r') as f:
    header = f.readline().strip()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            user_id, item_id, timestamp = parts[0], parts[1], parts[2]
            interactions.append((user_id, item_id, timestamp))

print(f"Original: {len(interactions)} interactions")
print(f"Header: {header}")

# Iterative 5-core filter
prev_count = -1
while prev_count != len(interactions):
    prev_count = len(interactions)
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for u, i, _ in interactions:
        user_count[u] += 1
        item_count[i] += 1
    
    filtered = [(u, i, t) for u, i, t in interactions 
                if user_count[u] >= MIN_INTER and item_count[i] >= MIN_INTER]
    interactions = filtered
    print(f"After filter: {len(interactions)} interactions")

# Stats
users = set(u for u, _, _ in interactions)
items = set(i for _, i, _ in interactions)
print(f"\nFinal 5-core dataset:")
print(f"  Users:        {len(users)}")
print(f"  Items:        {len(items)}")
print(f"  Interactions: {len(interactions)}")
print(f"  Avg seq len:  {len(interactions)/len(users):.2f}")

# Paper target:
print(f"\nPaper target:")
print(f"  Users:        27530")
print(f"  Items:        11260")
print(f"  Interactions: 231317")
print(f"  Avg seq len:  8.4")

# Write output
with open(OUTPUT, 'w') as f:
    f.write(header + '\n')
    for u, i, t in interactions:
        f.write(f"{u}\t{i}\t{t}\n")

print(f"\n✓ Written to {OUTPUT}")
