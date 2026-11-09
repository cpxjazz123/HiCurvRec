"""q9: one sampling path, full report, ordered identity, immutable run dir."""
from __future__ import annotations

import hashlib, json, time
from pathlib import Path

import torch

E = 16
DEFAULT_CKPT = Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/rqvae_best.pth")
SID = Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/item_sids.json")


def sha_bytes(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def triples_digest(triples) -> str:
    blob = json.dumps([[int(a), int(b), int(c), str(k)] for c, a, b, k in triples],
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def validate_identity(report) -> bool:
    """The consumer-side invariant: rows, list length and digest must agree."""
    ident = report["identity"]
    triples = [(int(c), int(a), int(b), str(k)) for c, a, b, k in ident["triples"]]
    if ident["z_struct_rows"] != 3 * ident["triples_accepted"]:
        raise ValueError("z_struct_rows must be 3*triples_accepted")
    if len(triples) != ident["triples_accepted"]:
        raise ValueError("the stored triple list must have triples_accepted entries")
    if ident["triples_sha256"] != triples_digest(triples):
        raise ValueError("the stored digest does not match the stored triples")
    return True


def split_latent(g):
    return {"E_norm": float(g[..., :E].pow(2).sum()) ** 0.5,
            "H_norm": float(g[..., E:].pow(2).sum()) ** 0.5}


def report_blocks(blocks, grads):
    out = {}
    for name, params in blocks.items():
        gs = grads.get(name, [None] * len(params))
        none = sum(1 for g in gs if g is None)
        zero = sum(1 for g in gs if g is not None and float(g.abs().sum()) == 0.0)
        nonzero = sum(1 for g in gs if g is not None and float(g.abs().sum()) > 0.0)
        flat = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for p, g in zip(params, gs)])
        out[name] = {"tensors": len(params), "none": none, "zero": zero,
                     "nonzero": nonzero, "norm": float(flat.norm())}
    return out


def _unique_dir(root: Path) -> Path:
    base = int(time.time())
    for offset in range(1000):
        candidate = root / ("run-%d" % (base + offset))
        if not candidate.exists():
            return candidate
    raise RuntimeError("no free run directory")


def run(out_root: Path, ckpt: Path | None = None) -> dict:
    import random
    import numpy as np
    import pandas as pd
    import mixed_geometry as G
    import structure as S
    import train_rqvae as T
    import training_hooks as H
    from model.model import RQVAE

    ckpt = Path(ckpt or DEFAULT_CKPT)
    seed = T.SEED
    requested, batch_n = 64, 64
    scale, lam = 0.04444597661495209, 9.226019983543665e-07
    torch.manual_seed(seed)
    emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, seed)
    items = torch.from_numpy(emb).float()
    model = RQVAE(T._tokenizer_config(), in_dim=emb.shape[1])
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit("checkpoint mismatch: missing=%s unexpected=%s"
                         % (missing[:3], unexpected[:3]))
    frame = pd.read_parquet(T.TRAIN_FILE)
    ids = np.unique(frame["target"].to_numpy(dtype=np.int64))[:batch_n]
    batch = items[torch.from_numpy(ids)]
    graph = S.build_graph(T.TRAIN_FILE, seed=seed)

    used = S.sample_triples(graph, requested, random.Random(seed))
    kinds = [k for *_, k in used]
    item_ids = [it for c, a, b, _ in used for it in (c, a, b)]

    captures = {"base": [], "struct": []}
    current = ["base"]

    def hook(module, inputs, output):
        output.retain_grad()
        captures[current[0]].append(output)

    handle = model.encoder.register_forward_hook(hook)
    try:
        current[0] = "base"
        recon, quant, _, _ = model.forward(batch)
        recon_loss = torch.nn.functional.mse_loss(recon, batch)
        base = model.compute_loss(batch, recon, quant)[0]
        current[0] = "struct"
        term = H.structure_term(model, items, graph, random.Random(seed), scale=scale,
                                size=len(used), triples=used,
                                behaviour_on=bool(T.BEHAVIOUR_ON))
        struct = lam * term
        total = base + struct
        z_base = captures["base"][0]
        z_struct = captures["struct"][0] if captures["struct"] else None

        blocks = {"encoder": list(model.encoder.parameters())}
        for i, layer in enumerate(model.rq.vq_layers):
            blocks[f"C{i+1}"] = [layer.get_code_embs()]
        blocks["decoder"] = list(model.decoder.parameters())
        flat = [p for ps in blocks.values() for p in ps]
        states, latent = {}, {}
        for name, loss in (("recon", recon_loss), ("quant", quant),
                           ("structure", struct), ("total", total)):
            raw = torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
            per, cur = {}, 0
            for block, ps in blocks.items():
                per[block] = raw[cur:cur + len(ps)]
                cur += len(ps)
            states[name] = report_blocks(blocks, per)
            targets = [z_base] + ([z_struct] if z_struct is not None else [])
            got = torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
            gb = got[0]
            gs = got[1] if len(got) > 1 else None
            latent[name] = {"z_base": None if gb is None else split_latent(gb.detach()),
                            "z_struct": ("NOT_APPLICABLE" if z_struct is None else
                                         (None if gs is None else split_latent(gs.detach())))}
        tb = torch.autograd.grad(total, z_base, retain_graph=True)[0]
        bb = torch.autograd.grad(base, z_base, retain_graph=True)[0]
        if z_struct is not None:
            ts = torch.autograd.grad(total, z_struct, retain_graph=True)[0]
            ss = torch.autograd.grad(struct, z_struct, retain_graph=True)[0]
            struct_diff = float((ts - ss).abs().max())
        else:
            struct_diff = None
        report = {
            "identity": {
                "script_sha256": sha_bytes(Path(__file__)),
                "checkpoint_sha256": sha_bytes(ckpt),
                "sid_sha256": sha_bytes(SID),
                "train_sha256": sha_bytes(Path(T.TRAIN_FILE)),
                "embedding_sha256": sha_bytes(Path(T.EMBEDDING_FILE)),
                "source_sha256": {name: sha_bytes(Path(__file__).parent / name)
                                  for name in ("model/layers.py", "model/model.py",
                                               "mixed_geometry.py", "native_product.py",
                                               "structure.py", "training_hooks.py")},
                "s": scale, "lambda": lam, "seed": seed,
                "batch_ids": [int(i) for i in ids],
                "triples_requested": requested, "triples_accepted": len(used),
                "triples": [[int(c), int(a), int(b), str(k)] for c, a, b, k in used],
                "triples_sha256": triples_digest(used),
                "z_base_rows": z_base.shape[0], "z_struct_rows": z_struct.shape[0],
                "mode": "train", "behaviour_on": bool(T.BEHAVIOUR_ON),
                "geometry_on": bool(G.GEOMETRY_ON),
            },
            "losses": {"recon": float(recon_loss.detach()), "quant": float(quant.detach()),
                       "structure_weighted": float(struct.detach()), "total": float(total.detach())},
            "states": states, "latent": latent,
            "diffs": {"base_vs_total_z_base": float((tb - bb).abs().max()),
                      "struct_vs_total_z_struct": struct_diff},
        }
    finally:
        handle.remove()

    validate_identity(report)
    out_dir = _unique_dir(Path(out_root))
    out_dir.mkdir(parents=True)
    (out_dir / "factors.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q9")
    print(json.dumps(run(root), indent=1)[:400])
