#!/usr/bin/env python3
"""Find N where McAuley 5core (57439 users) reduces to paper target (27530)."""

from collections import defaultdict

INPUT = "/home/wlia0047/ar57/wenyu/GeneRec/data/recbole/Musical_Instruments/Musical_Instruments.inter"

# Read interactions
interactions = []
with open(INPUT, 'r') as f:
    header = f.readline().strip()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            interactions.append((parts[0], parts[1], parts[2]))

print(f"Start: {len(interactions)} inter")
print()

for N in [5, 10, 15, 20, 25, 30, 40, 50]:
    data = list(interactions)
    prev = -1
    while prev != len(data):
        prev = len(data)
        u_cnt = defaultdict(int)
        i_cnt = defaultdict(int)
        for u, i, _ in data:
            u_cnt[u] += 1
            i_cnt[i] += 1
        data = [(u, i, t) for u, i, t in data if u_cnt[u] >= N and i_cnt[i] >= N]
    users = len(set(u for u, _, _ in data))
    items = len(set(i for _, i, _ in data))
    print(f"N={N}-core: users={users}, items={items}, inter={len(data)}, avg={len(data)/users:.2f}")
