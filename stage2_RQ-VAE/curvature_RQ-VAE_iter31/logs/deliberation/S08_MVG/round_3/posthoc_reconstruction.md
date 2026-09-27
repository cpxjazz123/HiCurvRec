# S08 post-hoc batch provenance reconstruction

Authorized by the S08 round-3 Judge for one CPU/data-only replay. Exact command: `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 /tmp/s08_iter31_provenance.py` (tool-reported elapsed time: 2.76 s). The temporary script read only the configured embedding header, item-ID sidecar, training parquet, and current selection/source code; it did not import the model/trainer/MVG checker, access a checkpoint, query CUDA/GPU, compute embeddings, or run model/gradient/training code.

Reconstructed 640 ordered source/future item-ID pairs with seed 42. Checks matched recorded shape `[24587, 768]`, 339519 transitions, 24474 active sources, and batch shape expectation `[640, 32]`. The full pair list, current input SHA-256 hashes, current source SHA-256 hashes, method, and flags are in `posthoc_batch_provenance.json`.

**Identity limit:** no pre-run input hashes were captured. These are post-hoc hashes of current bytes; they do not establish historical byte identity. The reconstructed pair list is conditional on current inputs matching the bytes used by the sole MVG run; it is not directly captured runtime provenance.

**Device limit:** no GPU/device query was made. Current source requests logical `cuda:0`; the original MVG output lacks a printed runtime device or physical GPU identity. This reconstruction cannot recover it.

No downstream stage is authorized by the reconstruction. Fresh independent S08 round-4 A/B assessments and Judge C adjudication are required before S08 can pass.
