"""CPU probe of the structure/reconstruction gradients, with explicit coverage flags.

Corrections applied after review:
  * it uses the tree's own model builder, weight init and KMeans codebook init with
    the frozen settings, and a batch drawn by the trainer's own rule -- it no
    longer constructs a bare RQVAE and calls it "the initialised model";
  * geometry-on and geometry-off are separate, explicitly labelled measurements;
  * reconstruction, quantisation and structure gradients are reported separately,
    so "the encoder still gets gradient from the reconstruction loss alone" can be
    read directly;
  * pairs are chosen by the FULL three-token SID: a different-item pair that shares
    the whole SID, and a pair whose SIDs differ.  When a category does not exist
    the probe reports NOT_COVERED and the reason instead of substituting a pair;
  * the scale is the frozen TRAIN calibration, not a per-pair function of the
    distance being measured.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_geometry as G
import structure as S
import training_hooks as H
import train_rqvae as T

SK_ITERS = 50            # frozen baseline value, not the probe's 5
SK_EPSILON = 0.003
PROBE_ITEMS = 256


def build_model_and_batch():
    embeddings = torch.from_numpy(T.load_embeddings(T.EMBEDDING_FILE)).float()
    frame = T.pd.read_parquet(T.TRAIN_FILE)
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    all_items = embeddings
    train_items = embeddings[torch.from_numpy(train_ids)]
    config = T._tokenizer_config()
    config.sk_iters = SK_ITERS
    config.sk_epsilon = SK_EPSILON
    model = T.RQVAE(config, in_dim=embeddings.shape[1])
    if T.XAVIER_INIT:
        T.initialize_tiger_weights(model)
    model.init_codebook(train_items)
    model.eval()
    batch = train_items[:PROBE_ITEMS]
    return model, all_items, batch


def classify_pairs(tokens: torch.Tensor):
    """(a, b, kind) by full three-token SID, or a NOT_COVERED record with the reason."""
    rows = [tuple(int(x) for x in row) for row in tokens.tolist()]
    same = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows)) if rows[i] == rows[j]]
    different = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows)) if rows[i] != rows[j]]
    out = {}
    out["shared_sid"] = (same[0] + ("shared_sid",)) if same else None
    out["different_sid"] = (different[0] + ("different_sid",)) if different else None
    out["not_covered"] = {name: ("no pair with this property among %d items" % len(rows))
                          for name, value in out.items() if name != "not_covered" and value is None}
    out["tokens"] = rows
    return out


def grad_report(term, model, blocks):
    report = {}
    for name, params in blocks.items():
        params = [p for p in params if p.requires_grad]
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        report[name] = float(sum(float(g.pow(2).sum()) for g in grads if g is not None)) ** 0.5
    return report


def main():
    model, all_items, batch = build_model_and_batch()
    with torch.no_grad():
        _, _, _, tokens, levels = model.forward_with_codes(batch)
    pairs = classify_pairs(tokens)
    blocks = {"encoder": list(model.encoder.parameters()),
              "codebooks": [p for layer in model.rq.vq_layers for p in layer.parameters()]}

    graph = S.build_graph(T.TRAIN_FILE, seed=T.SEED)
    scale = H.calibrate_structure_scale(model, all_items, graph, random.Random(T.SEED),
                                        size=T.STRUCT_CALIBRATION_TRIPLES)
    out = {"scale": scale, "coverage": {k: pairs[k] for k in ("shared_sid", "different_sid", "not_covered")}}

    for kind in ("shared_sid", "different_sid"):
        record = pairs[kind]
        if record is None:
            out[kind] = "NOT_COVERED"
            continue
        a, b, _ = record
        for mode in ("geometry_on", "geometry_off"):
            geometry = mode == "geometry_on"
            _, _, _, pair_tokens, pair_levels = model.forward_with_codes(batch[[a, b]])
            positions = pair_levels[-1] + (model.codes_to_positions(pair_tokens)
                                           - model.codes_to_positions(pair_tokens).detach())
            structure_term = S.pair_stress(positions[0:1], positions[1:2], scale=scale,
                                           h=1.0, geometry_on=geometry)
            reconstructed, quant_loss, _, _, _ = model.forward_with_codes(batch[[a, b]])
            recon_only = torch.nn.functional.mse_loss(reconstructed, batch[[a, b]])
            combined, _ = model.compute_loss(batch[[a, b]], reconstructed, quant_loss)
            out.setdefault(kind, {})[mode] = {
                "distance_sq": float(G.mixed_sq(positions[0:1], positions[1:2], geometry)),
                "structure_term": float(structure_term),
                "grad_structure": grad_report(structure_term, model, blocks),
                "grad_reconstruction_only": grad_report(recon_only, model, blocks),
                "grad_reconstruction_plus_quant": grad_report(combined, model, blocks),
            }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
