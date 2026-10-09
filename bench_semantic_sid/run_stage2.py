"""Stage2-only real-T5 semantic SID experiment. No Stage3.

Required: data/semantic_item_hierarchy.npz containing coarse, fine arrays,
both aligned row-for-row to stage1_GeneEmbedding/output/sentence_t5.npy.
Unlabeled items use -1 for both fields. Each fine class has one coarse parent.
This taxonomy is NOT currently supplied in the repository: fail closed.
Run on a GPU: python -m bench_semantic_sid.run_stage2
"""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_mutual_info_score

from bench_hier_behavior.models import RQVAE
from .hierarchy_losses import SemanticHierarchyLoss

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/semantic_item_hierarchy.npz"
FEATURES = ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
OUT = ROOT / "results/bench_semantic_sid"
ARMS = ("euclid_rq", "hyp_rq", "euclid_cone_prefix", "hyp_cone_prefix")
SEEDS = (42, 43, 44)
STEPS, BATCH_SIZE, SUPERVISED_BATCH = 1200, 256, 128
CONELAMBDA, PREFIXLAMBDA = 0.05, 0.01


def load():
    if not DATA.is_file():
        raise FileNotFoundError(
            f"Required VERIFIED semantic category artifact absent: {DATA}; "
            "do not substitute co-purchase clusters or model-generated taxonomy.")
    if not FEATURES.is_file():
        raise FileNotFoundError(str(FEATURES))
    X = np.load(FEATURES).astype("float32", copy=False)
    with np.load(DATA) as z:
        a, b = np.asarray(z["coarse"], dtype="int64"), np.asarray(z["fine"], dtype="int64")
    if not (len(X) == len(a) == len(b)) or not np.isfinite(X).all():
        raise ValueError("T5 and category data must align and be finite")
    if np.any((a < 0) != (b < 0)):
        raise ValueError("Missing-category rows must have both fields -1")
    ids = np.flatnonzero((a >= 0) & (b >= 0))
    if len(ids) < 100:
        raise ValueError("Require >=100 verified taxonomy item labels")
    _, coarse = np.unique(a[ids], return_inverse=True)
    _, fine = np.unique(b[ids], return_inverse=True)
    ncoarse, nfine = len(np.unique(coarse)), len(np.unique(fine))
    if ncoarse < 2 or nfine < 4:
        raise ValueError("Need >=2 coarse and >=4 fine groups")
    parent = np.full(nfine, -1, dtype="int64")
    for c, f in zip(coarse, fine):
        if parent[f] not in (-1, c):
            raise ValueError("A fine label is assigned to two coarse classes")
        parent[f] = c
    return X, ids, coarse, fine, parent


def split(fine, rng):
    fit, hold = [], []
    for f in np.unique(fine):
        members = np.flatnonzero(fine == f)
        rng.shuffle(members)
        count = max(1, round(0.2*len(members))) if len(members) >= 5 else 0
        hold.extend(members[:count])
        fit.extend(members[count:])
    if not hold:
        raise ValueError("Insufficient examples for category-stratified holdout")
    return np.asarray(fit), np.asarray(hold)


def supervised_indices(coarse, fine, fit, rng):
    buckets = {}
    for idx in fit:
        buckets.setdefault(int(coarse[idx]), {}).setdefault(int(fine[idx]), []).append(int(idx))
    selected = []
    roots = list(buckets)
    for c in rng.choice(roots, 4, replace=len(roots)<4):
        children = list(buckets[int(c)])
        for f in rng.choice(children, 2, replace=len(children)<2):
            selected.extend(rng.choice(buckets[int(c)][int(f)], SUPERVISED_BATCH//8,
                                       replace=True).tolist())
    return np.asarray(selected, dtype="int64")


def metrics(model, X, ids, coarse, fine, hold):
    model.eval()
    with torch.no_grad():
        latent = model._training_latent(X)
        _, _, tokens = model.rq(latent)
    sid = tokens.cpu().numpy()
    chosen = sid[ids[hold]]
    two = chosen[:,0].astype("int64")*256 + chosen[:,1].astype("int64")
    return {
        "l1_coarse_ami": float(adjusted_mutual_info_score(coarse[hold], chosen[:,0])),
        "l12_fine_ami": float(adjusted_mutual_info_score(fine[hold], two)),
        "full_sid_collision_fraction": float(1 - len(np.unique(sid,axis=0))/len(sid)),
        "unique_sids": int(len(np.unique(sid,axis=0))),
        "holdout_count": int(len(hold)),
    }


def main():
    X, ids, coarse, fine, parent = load()
    if not torch.cuda.is_available():
        raise RuntimeError("Use GPU for the real Stage2 benchmark")
    OUT.mkdir(parents=True, exist_ok=True)
    x = torch.from_numpy(X).to("cuda")
    for seed in SEEDS:
        fit, hold = split(fine, np.random.default_rng(seed+953))
        for arm in ARMS:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed+112)
            supervised = arm.endswith("_cone_prefix")
            model = RQVAE(
                "poincare" if arm.startswith("hyp_") else "euclid",
                in_dim=X.shape[1], codebook_dim=32, normalize_latent=True,
                hidden_sizes=(512, 256, 128),
            ).to("cuda")
            assert model.rq.codebook_sizes == (256,256,256)
            model.init_codebooks(x, seed)
            loss_fn = (SemanticHierarchyLoss(len(np.unique(coarse)),
                       torch.from_numpy(parent), 32).to("cuda") if supervised else None)
            parameters = list(model.parameters()) + (list(loss_fn.parameters()) if supervised else [])
            optimizer = torch.optim.AdamW(parameters, lr=1e-3)
            model.train()
            for step in range(STEPS):
                rows = rng.choice(len(X), size=BATCH_SIZE, replace=True)
                batch = x[torch.from_numpy(rows).to("cuda")]
                recon, vq, _, _ = model(batch)
                total = F.mse_loss(recon, batch) + vq
                if supervised:
                    rows = supervised_indices(coarse, fine, fit, rng)
                    select = torch.from_numpy(ids[rows]).to("cuda")
                    _, _, tok, latent, prefixes = model(x[select],return_prefixes=True)
                    parts = loss_fn(model,latent,tok,prefixes,
                        torch.from_numpy(coarse[rows]).to("cuda"),
                        torch.from_numpy(fine[rows]).to("cuda"))
                    total = total + CONELAMBDA*parts["cone"] + PREFIXLAMBDA*parts["prefix"]
                if not bool(torch.isfinite(total.detach())):
                    raise RuntimeError(f"Nonfinite {arm} seed {seed} step {step}")
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
            result = {"arm": arm, "seed":seed, "steps":STEPS,
                      "input_dim":X.shape[1], "latent_dim":32,
                      "cone_weight":CONELAMBDA if supervised else 0.,
                      "prefix_weight":PREFIXLAMBDA if supervised else 0.,
                      "fit_items":int(len(fit)), "metrics":metrics(model,x,ids,coarse,fine,hold)}
            out = OUT / f"{arm}_seed{seed}"
            out.mkdir(exist_ok=True)
            (out/"metrics.json").write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True)
    print("Stage2 benchmark finished. No Stage3 training has been run.")


if __name__ == "__main__":
    main()
