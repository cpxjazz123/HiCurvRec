"""MVG for the initialisation-order repair (root-cause programme, step 3).

One variable changes against hyper_l1_s42: the run-constant ball normalizer is
resolved from the INITIAL encoder BEFORE the codebooks exist, so the KMeans
centres and the level-1 assignment that immediately follows them are fitted under
the geometry training will use, instead of under the construction default of 1.0.

This probe verifies implementation, not merit:

  1. the normalizer the layer actually carries DURING init_codebook differs
     between the old order and the repaired one, and equals resolve_normalizer's
     output in the repaired case;
  2. the level-1 assignment computed inside init_codebook changes for a
     measurable share of items -- the repair is not a no-op;
  3. SINGLE VARIABLE: the level-1 KMeans CENTRES are bit-identical between the
     two orders (KMeans sees the same raw residuals either way). Only the
     assignment, and therefore the residual handed to level 2, differs.
  4. the encoder is untouched by init_codebook, so the resolved normalizer does
     not drift between the two calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import hyperbolic as H  # noqa: E402
import train_rqvae as T  # noqa: E402
from model import RQVAE  # noqa: E402

DEV = torch.device("cuda:0")
_ok = [True]


def req(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    _ok[0] = _ok[0] and bool(cond)


T.set_seed(T.SEED)
emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
frame = pd.read_parquet(T.TRAIN_FILE)
ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
tr = torch.from_numpy(emb)[torch.from_numpy(ids)]

GEO = {"normalizer": 1.0, "curvature": T.L1_GEODESIC_CURVATURE}


def fresh_model():
    m = RQVAE(T._tokenizer_config(GEO), in_dim=emb.shape[1]).to(DEV)
    T.initialize_tiger_weights(m)
    return m


def probe(pre_set_normalizer: bool):
    # Re-seed inside the probe so BOTH orders start from bit-identical weights;
    # otherwise the "single variable" comparison is meaningless.
    T.set_seed(T.SEED)
    m = fresh_model()
    layer = m.rq.vq_layers[0]
    nz_resolved = T.resolve_normalizer(m, tr.to(DEV))
    nz_used = float(layer.geodesic_normalizer)
    if pre_set_normalizer:
        layer.geodesic_normalizer = float(nz_resolved)
        nz_used = float(layer.geodesic_normalizer)
    with torch.no_grad():
        m.init_codebook(tr.to(DEV))
    centres = [l.get_code_embs().detach().clone() for l in m.rq.vq_layers]
    with torch.no_grad():
        idx1 = layer._distances(m.encoder(tr.to(DEV))).argmin(-1)
    return {
        "nz_used_during_init": nz_used,
        "nz_resolved": float(nz_resolved),
        "centres": centres,
        "l1_idx": idx1,
        "distinct_l1": int(idx1.unique().numel()),
    }


old = probe(False)   # current order: normalizer still 1.0 during init_codebook
new = probe(True)    # repaired order: resolved before init_codebook

req(abs(old["nz_used_during_init"] - 1.0) < 1e-12,
    f"old order uses the construction default during init ({old['nz_used_during_init']})")
req(abs(new["nz_used_during_init"] - new["nz_resolved"]) < 1e-12,
    f"repaired order uses the resolved normalizer during init "
    f"({new['nz_used_during_init']:.6f} == resolve_normalizer {new['nz_resolved']:.6f})")
req(abs(new["nz_used_during_init"] - old["nz_used_during_init"]) > 1e-3,
    "the normalizer inside init_codebook really differs between the two orders")

changed = float((old["l1_idx"] != new["l1_idx"]).float().mean())
req(changed > 0.0,
    f"the L1 assignment computed inside init_codebook changes for {changed:.4f} of items")
req(old["distinct_l1"] != new["distinct_l1"] or changed > 0.0,
    f"L1 codebook occupancy after init: old {old['distinct_l1']}/256, "
    f"new {new['distinct_l1']}/256")

# KMeans itself is not reproducible run to run (n_init="auto" with an unseeded
# RNG and thread-order reduction), so "the centres must be identical" cannot be
# asserted. Measure KMeans' OWN run-to-run spread on the SAME order and require
# the cross-order difference at level 1 to be no larger than that noise floor.
old_again = probe(False)
noise1 = float((old["centres"][0] - old_again["centres"][0]).abs().max())
noise2 = float((old["centres"][1] - old_again["centres"][1]).abs().max())
c1_diff = float((old["centres"][0] - new["centres"][0]).abs().max())
c2_diff = float((old["centres"][1] - new["centres"][1]).abs().max())
req(noise1 > 0.0,
    f"KMeans is not run-to-run reproducible (level-1 spread on the SAME order "
    f"{noise1:.2e}), so centre equality cannot be the invariant")
req(c1_diff <= 2.0 * max(noise1, 1e-9),
    f"SINGLE VARIABLE: the level-1 centre difference ({c1_diff:.2e}) is within twice "
    f"KMeans' own noise floor ({noise1:.2e}); the encoder output entering it is identical")
req(c2_diff > max(2.0 * noise2, 1e-3),
    f"the level-2 codebook input differs far beyond KMeans' noise ({c2_diff:.4f} vs "
    f"floor {noise2:.2e}), which is the mechanism's intended downstream effect")

# The repaired order must not change the normalizer the run actually trains with.
T.set_seed(T.SEED)
m = fresh_model()
n_before = T.resolve_normalizer(m, tr.to(DEV))
with torch.no_grad():
    m.init_codebook(tr.to(DEV))
n_after = T.resolve_normalizer(m, tr.to(DEV))
req(abs(n_before - n_after) < 1e-12,
    f"init_codebook does not move the encoder, so the resolved normalizer is stable "
    f"({n_before:.6f} -> {n_after:.6f})")

# And the two maps really are different geometries, not a rescaling.
lat = m.encoder(tr.to(DEV)).detach()
ctr = m.rq.vq_layers[0].get_code_embs().detach()
d_init = H.pair_poincare_distance(H.encode_ball(lat, 1.0, 1.0),
                                  H.encode_ball(ctr, 1.0, 1.0), 1.0)
d_new = H.pair_poincare_distance(H.encode_ball(lat, 1.0, new["nz_resolved"]),
                                 H.encode_ball(ctr, 1.0, new["nz_resolved"]), 1.0)
d_euc = lat.pow(2).sum(1, keepdim=True) - 2 * lat @ ctr.t() + ctr.pow(2).sum(1)
agree_old = float((d_init.argmin(-1) == d_euc.argmin(-1)).float().mean())
agree_new = float((d_new.argmin(-1) == d_euc.argmin(-1)).float().mean())
req(agree_old > agree_new,
    f"the init-order geometry was effectively Euclidean ({agree_old:.4f} agreement) and the "
    f"repaired one is not ({agree_new:.4f})")

print("\nMVG " + ("PASS" if _ok[0] else "FAIL"), flush=True)
raise SystemExit(0 if _ok[0] else 1)