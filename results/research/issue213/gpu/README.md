# ORCA `node 5` on the GPU — what was measured, and how to rebuild it

`orca node 5` is the tool behind every five-node structural feature in this directory. It took
**951.18 s** on the TRAIN behaviour graph (24,587 nodes, 315,391 edges, max degree 2,220). The
profile says where that goes, and this directory holds the two accelerations that were measured.

| version | wall clock | vs single-threaded | output |
|---|---|---|---|
| `orca node 5` (stock) | 951.18 s | 1× | ground truth |
| `orca_omp.cpp` — OpenMP over stages 2 and 3 | 142.01 s | 6.7× | byte-identical |
| `orca_gpu.cu` — stage 1 on the GPU, stages 2–3 on 32 cores | **34.02 s** | **28×** | **byte-identical** |

Stage wall clock of the OpenMP build, measured with `omp_get_wtime()`:

| stage | CPU | wall (32 threads) |
|---|---|---|
| 1 — precompute `common2`/`common3` | 84.15 s | **84.75 s** (serial: this is the wall) |
| 2 — count full graphlets | 0.74 s | 0.04 s |
| 3 — build the systems of equations | 1,047 s | 45.0 s (23×) |

`perf record` on 70 s of the stock run: `adjacent_matrix` 39.7%, `common2` hash find 20.1%,
`count5` body 16.5%, `common3` hash find 15.2% — memory-latency bound, which is why more threads
stop helping once stage 1 is left serial (`-O3 -march=native` changed nothing: 141.96 s).

## Where the GPU pays

`build_common_bench.cu` walks stage 1's enumeration alone: **8.66e9 triple visits in 0.883 s**
(9.80 G visits/s) against 84.75 s on one core, and only **58.1M** of those visits pass the
two-internal-edge test, so the tables are small — `common2` 25,430,338 entries,
`common3` 15,347,954 entries.

`orca_gpu.cu` builds both on the device (atomic `common2` over a flat `n*n` array, open-addressing
`common3`), copies them to the host, and leaves stages 2–3 on the CPU against flat lookups instead
of `unordered_map`, which is why stage 3's CPU time fell from 1,047 s to 655 s as well.

Two bugs cost a full debugging cycle each, both found by diffing the tables key by key against the
CPU build, never by looking at the orbit output:

1. **Lost CAS race.** After a failed `atomicCAS` the probe moved on, so a key could be written into
   two slots and every lookup would see one of them; counts came out slightly low.
2. **`(key * K) & mask` is not a hash.** The low 27 bits of the product depend only on the low 27
   bits of the key, and the key packs `a<<42 | b<<21 | c`, so every triple sharing `(b mod 64, c)`
   landed in one slot, probe chains passed the bound and **1,494,234 triples (9.7%) were dropped**.
   The entry counts said it exactly: 13,853,720 against 15,347,954. A murmur3 finaliser before the
   mask fixed it, and the table then matched the CPU row for row.

## Rebuild

```bash
cd results/research/issue213/orca
git clone --depth 1 https://github.com/thocevar/orca . && g++ -O2 -std=c++11 -o orca orca.cpp
python3 ../gpu/make_variants.py                       # writes orca_omp.cpp and orca_gpu.cu
g++ -O2 -fopenmp -o orca_omp orca_omp.cpp
/usr/local/cuda-12.5/bin/nvcc -O3 -std=c++11 -Xcompiler -fopenmp -o orca_gpu orca_gpu.cu
OMP_NUM_THREADS=32 ./orca_omp node 5 graph.in out_omp.out   # 142 s
./orca_gpu node 5 graph.in out_gpu.out                      # 34 s
cmp node5.out out_omp.out && cmp node5.out out_gpu.out      # both must be silent
```

## What is still on the CPU

Stage 3 — 45 s wall of the OpenMP build, ~25 s of the 34 s here. Moving it to the device is
possible: `common_x[b]` in that code equals `common2(x,b)` exactly, so the per-node scratch that
blocks a one-thread-per-node port can be dropped, and hubs need block-per-node with atomics. That
is a rewrite, not a flag, and it has not been started.
