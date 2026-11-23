"""Run the quantized-geometry fidelity evaluation and persist the report (issue #249, step 6).

The evaluation uses the **checkpoint-consistent** TIGER pair (that checkpoint and the table it
produced). The Stage3 table has no same-source codebook, so its geometry metrics are NA; the pure
SID metrics are recomputed for it separately.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch

from tools.graph_infra import fidelity, quant_eval, tiger_compare
from tools.graph_infra.assets import load_sparse
from tools.graph_infra.curvature_stream import CurvatureReader

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / 'assets'
OUT = REPO / 'results/research/issue249/FIDELITY.json'
TRIPLES = 2000


def main() -> int:
    start = time.monotonic()
    reader = CurvatureReader(REPO, ASSETS)
    node_moments = dict(np.load(ASSETS / 'node_moments.npz'))
    undirected = load_sparse(ASSETS / 'train/visible_undirected.npz')
    checkpoint = REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth'
    reexport = REPO / 'results/research/infra246/tiger_reexport/item_sids.json'
    codebooks = quant_eval.load_codebooks(checkpoint)
    sid = quant_eval.load_sid_table(reexport)
    q = fidelity.decode_hard_sid(sid, codebooks)
    z = quant_eval.load_continuous_z(checkpoint)
    triples = quant_eval.sample_triples(reader, TRIPLES)
    report = quant_eval.evaluate(triples, reader, node_moments, undirected, q, z)
    report['provenance'] = {
        'note': ('the checkpoint-consistent pair only; the Stage3 table has no same-source '
                 'codebook, so its geometry metrics are NA'),
        'checkpoint': str(checkpoint.relative_to(REPO)),
        'checkpoint_sha256': tiger_compare.sha256(checkpoint),
        'sid_table': str(reexport.relative_to(REPO)),
        'codebook_levels': [int(level.shape[0]) for level in codebooks],
        'codebook_dim': int(codebooks[0].shape[1]),
        'triples_sampled': int(len(triples)),
        'sampling': 'deterministic, seed 249002, spread over distinct centres',
        'reference_set': 'the centre component, the same indices on both sides',
    }
    stage3 = tiger_compare.compare(REPO)
    report['stage3_table'] = {
        'sid_sha256': stage3['sid_sha256'],
        'pure_sid_metrics': stage3['pure_sid_metrics'],
        'geometry_metrics': stage3['geometry_metrics'],
    }
    report['seconds'] = time.monotonic() - start
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=float) + '\n')
    print(json.dumps({
        'triples_evaluated': report['triples_evaluated'],
        'frozen_target_matches': report['graph_side_recomputation_matches_the_frozen_target']['matches'],
        'max_abs_difference_vs_frozen': report['graph_side_recomputation_matches_the_frozen_target']['max_abs_difference'],
        'hard_sid_sign_agreement': report['hard_sid_decoded']['sign_agreement']['sign_agreement'],
        'continuous_z_sign_agreement': report['continuous_z']['sign_agreement']['sign_agreement'],
        'seconds': report['seconds'],
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
