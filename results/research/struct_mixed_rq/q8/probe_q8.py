"""q8 report: real JSON states plus two same-graph latent tensors.

Importing this module performs no measurement and writes nothing; the work happens
only in main().
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def report_blocks(blocks, grads) -> dict:
    """Keep parameter identity and the three gradient states in the report."""
    out = {}
    for name, params in blocks.items():
        gs = grads.get(name, [None] * len(params))
        none = sum(1 for g in gs if g is None)
        zero = sum(1 for g in gs if g is not None and float(g.abs().sum()) == 0.0)
        nonzero = sum(1 for g in gs if g is not None and float(g.abs().sum()) > 0.0)
        flat = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for p, g in zip(params, gs)])
        out[name] = {"tensors": len(params), "none_tensors": none,
                     "zero_tensors": zero, "nonzero_tensors": nonzero,
                     "norm": float(flat.norm())}
    return out


def latent_grads(model, batch, items, graph, triples, scale, lam, rng_seed, blocks):
    """Capture z_base and z_struct from the graph that produced the losses."""
    captures = {"base": [], "struct": []}

    def hook(module, inputs, output):
        output.retain_grad()
        captures[current[0]].append(output)

    current = ["base"]
    handle = model.encoder.register_forward_hook(hook)
    try:
        current[0] = "base"
        recon, quant, _, _ = model.forward(batch)
        base = model.compute_loss(batch, recon, quant)[0]
        current[0] = "struct"
        import structure as S
        import training_hooks as H
        term = H.structure_term(model, items, graph, __import__("random").Random(rng_seed),
                                scale=scale, size=len(triples))
        struct = lam * term
        total = base + struct
        z_base = captures["base"][0]
        z_struct = captures["struct"][0]
        model.zero_grad(set_to_none=True)
        flat = [p for ps in blocks.values() for p in ps]
        state = {}
        for name, loss in (("base", base), ("struct", struct), ("total", total)):
            raw = torch.autograd.grad(loss, flat, retain_graph=True, allow_unused=True)
            per, cur = {}, 0
            for block, ps in blocks.items():
                per[block] = raw[cur:cur + len(ps)]
                cur += len(ps)
            state[name] = report_blocks(blocks, per)
        pairs = {}
        for name, loss in (("base", base), ("struct", struct), ("total", total)):
            gb, gs = torch.autograd.grad(loss, [z_base, z_struct], retain_graph=True,
                                         allow_unused=True)
            pairs[name] = {
                "z_base": None if gb is None else gb.detach(),
                "z_struct": None if gs is None else gs.detach(),
            }
        return (pairs, state,
                {"z_base_rows": z_base.shape[0], "z_struct_rows": z_struct.shape[0],
                 "z_base_shape": list(z_base.shape),
                 "z_struct_shape": list(z_struct.shape)})
    finally:
        handle.remove()


def main(out_dir: Path) -> dict:
    import hashlib
    import numpy as np
    import pandas as pd
    import mixed_geometry as G
    import structure as S
    import train_rqvae as T
    import training_hooks as H
    from model.model import RQVAE

    seed = T.SEED
    batch_n, triples_n = 64, 64
    scale, lam = 0.04444597661495209, 9.226019983543665e-07
    ckpt = Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/rqvae_best.pth")
    torch.manual_seed(seed)
    emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, seed)
    items = torch.from_numpy(emb).float()
    model = RQVAE(T._tokenizer_config(), in_dim=emb.shape[1])
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: {missing[:3]} {unexpected[:3]}")
    frame = pd.read_parquet(T.TRAIN_FILE)
    ids = np.unique(frame["target"].to_numpy(dtype=np.int64))[:batch_n]
    batch = items[torch.from_numpy(ids)]
    graph = S.build_graph(T.TRAIN_FILE, seed=seed)
    triples = S.sample_triples(graph, triples_n, __import__("random").Random(seed))

    blocks = {"encoder": list(model.encoder.parameters())}
    for i, layer in enumerate(model.rq.vq_layers):
        blocks[f"C{i+1}"] = [layer.get_code_embs()]
    blocks["decoder"] = list(model.decoder.parameters())
    pairs, state, shapes = latent_grads(model, batch, items, graph, triples, scale, lam, seed, blocks)
    identity = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "sid_sha256": hashlib.sha256(Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/item_sids.json").read_bytes()).hexdigest(),
        "layers_sha256": hashlib.sha256(Path(__file__).with_name("model").joinpath("layers.py").read_bytes()).hexdigest(),
        "s": scale, "lambda": lam, "seed": seed,
        "batch_ids": [int(i) for i in ids], "triples_accepted": len(triples),
    }
    report = {"identity": identity, "shapes": shapes, "states": {}, "block_states": state,
              "diff_base_vs_total": {}, "diff_struct_vs_scaled_total": {}}
    tb = pairs["total"]["z_base"]; bb = pairs["base"]["z_base"]
    ts = pairs["total"]["z_struct"]; ss = pairs["struct"]["z_struct"]
    report["diff_base_vs_total"] = None if (tb is None or bb is None) else float((tb - bb).abs().max())
    report["diff_struct_vs_scaled_total"] = None if (ts is None or ss is None) else float((ts - ss).abs().max())
    report["states"] = {"z_base_grad_present": tb is not None,
                        "z_struct_grad_present": ts is not None}
    report["blocks_present"] = sorted(blocks)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "factors.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q8")
    print(json.dumps(main(out), indent=1)[:800])
