"""Gate 1 of issue #155: the two-arm differential fit (real AF3 targets versus random).

Three arms share one seed, one data order, the trainer's own optimiser and 200 fixed steps:
R with the real per-edge targets, N with the same targets randomly permuted across edges, and
U with the supervision switched off. Nothing is selected or tuned afterwards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import target_distance as td  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402

STEPS = 200
SIZE = 512
EVAL = 2048


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    table = json.loads((root / "results" / "research" / "issue155" / "af3_edges.json").read_text())
    dref = json.loads((root / "results" / "research" / "issue155" / "gate_dref.json").read_text())
    d_ref = float(dref["d_ref"])
    raw = np.load(root / "results" / "research" / "issue150" / "structure_prior.npz",
                  allow_pickle=True)
    item_of_node = {index: int(item) for index, item in enumerate(raw["item_ids"])}
    all_ids = sorted(item_of_node.values())
    # vectorised edge arrays: the per-step Python loop over 315k edges was the bottleneck
    edge_rows, edge_u = [], []
    for (tail, head), u in zip(table["edges"], table["u"]):
        left, right = item_of_node.get(int(tail)), item_of_node.get(int(head))
        if left is None or right is None:
            continue
        edge_rows.append((left, right))
        edge_u.append(float(u))
    edge_items = np.asarray(edge_rows, dtype=np.int64)
    edge_targets = np.asarray(edge_u, dtype=np.float64)
    top_item = int(max(edge_items.max(), max(all_ids))) + 1
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))

    def fresh() -> MixedRQVAE:
        torch.manual_seed(trainer.SEED)
        model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
        trainer.initialize_tiger_weights(model)
        model.init_codebook(rows[torch.from_numpy(np.asarray(all_ids[:4096]))])
        return model

    eval_ids = np.asarray(all_ids)[:EVAL]
    position_eval = {int(item): index for index, item in enumerate(eval_ids)}
    eval_mask = np.array([left in position_eval and right in position_eval
                          for left, right in edge_items], dtype=bool)
    eval_pairs = edge_items[eval_mask]
    eval_u = edge_targets[eval_mask]
    eval_edges = [(position_eval[int(left)], position_eval[int(right)])
                  for left, right in eval_pairs]
    eval_edges_t = torch.tensor(eval_edges, dtype=torch.long) if eval_edges else torch.zeros(0, 2, dtype=torch.long)
    eval_u_t = np.asarray(eval_u)

    def rank_consistency(model) -> float:
        model.eval()
        model(rows[torch.from_numpy(eval_ids)])
        prefixes = model.rq.last_prefixes
        distance = td.edge_distance(prefixes[-1][0].detach(), prefixes[-1][1].detach(),
                                    eval_edges_t, 1.0).numpy()
        return float(spearmanr(eval_u_t, distance).correlation)

    def codes(model) -> torch.Tensor:
        model.eval()
        model(rows[torch.from_numpy(eval_ids)])
        return model.rq.last_codes[-1].clone()

    _fitted_model: list = []

    def fit(mode: str) -> dict:
        model = fresh()
        optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.LR,
                                      weight_decay=trainer.WEIGHT_DECAY)
        generator = torch.Generator().manual_seed(15520)
        model.train()
        for step in range(STEPS):
            pick = torch.randperm(len(all_ids), generator=generator)[:SIZE].numpy()
            batch_ids = np.asarray(all_ids)[pick]
            batch = rows[torch.from_numpy(batch_ids)]
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
            loss, _ = model.compute_loss(batch, reconstructed, quant_loss)
            if mode != "U":
                position = {int(item): index for index, item in enumerate(batch_ids)}
                inside = np.zeros(top_item, dtype=bool)
                inside[batch_ids] = True
                mask = inside[edge_items[:, 0]] & inside[edge_items[:, 1]]
                pairs = [(position[int(a)], position[int(b)]) for a, b in edge_items[mask]]
                targets = edge_targets[mask].tolist()
                if pairs:
                    noise = torch.Generator().manual_seed(15521 + step)
                    if mode == "N":
                        targets = list(np.random.default_rng(15521 + step).permutation(targets))
                    terms = td.structure_losses(
                        model.rq.last_prefixes, torch.tensor(pairs, dtype=torch.long),
                        td.target_distance(torch.tensor(targets, dtype=torch.float64), d_ref),
                        torch.zeros(0, 3, dtype=torch.long), torch.tensor(d_ref), 1.0)
                    loss = loss + terms["struct"] + terms["tri"]
            loss.backward()
            optimizer.step()
        _fitted_model.clear(); _fitted_model.append(model)
        return {"rank_consistency": rank_consistency(model), "codes": codes(model).tolist()}

    report = {"steps": STEPS, "batch": SIZE, "eval_rows": EVAL, "d_ref": d_ref,
              "eval_edges": int(eval_edges_t.shape[0])}
    def gate3_cosine(model) -> dict:
        """Real-edge versus random-edge gradient direction, on the fitted model."""
        generator3 = torch.Generator().manual_seed(15599)
        pick3 = torch.randperm(len(all_ids), generator=generator3)[:SIZE].numpy()
        batch3 = rows[torch.from_numpy(np.asarray(all_ids)[pick3])]
        position3 = {int(item): index for index, item in enumerate(np.asarray(all_ids)[pick3])}
        inside3 = np.zeros(top_item, dtype=bool); inside3[np.asarray(all_ids)[pick3]] = True
        mask3 = inside3[edge_items[:, 0]] & inside3[edge_items[:, 1]]
        pairs3 = torch.tensor([(position3[int(a)], position3[int(b)]) for a, b in edge_items[mask3]],
                              dtype=torch.long)
        targets3 = td.target_distance(torch.tensor(edge_targets[mask3], dtype=torch.float64), d_ref)
        model.train(); model(batch3)
        grads = {}
        for name, edges_used in (("real", pairs3), ("random", torch.randint(0, SIZE, pairs3.shape, generator=generator3))):
            model.zero_grad(set_to_none=True)
            loss = td.structure_losses(model.rq.last_prefixes, edges_used, targets3,
                                       torch.zeros(0, 3, dtype=torch.long),
                                       torch.tensor(d_ref), 1.0)["struct"]
            loss.backward(retain_graph=True)
            grads[name] = torch.cat([p.grad.reshape(-1) for p in model.encoder.parameters()
                                     if p.grad is not None])
        return {"cosine": float(torch.nn.functional.cosine_similarity(grads["real"], grads["random"], dim=0)),
                "grad_norm_real": float(grads["real"].norm()),
                "grad_norm_random": float(grads["random"].norm())}
    out = {}
    for mode in ("R",):
        out[mode] = fit(mode)
        report["gate3_post_fit"] = gate3_cosine(_fitted_model[0])
        print(f"[arm {mode}] rank_consistency={out[mode]['rank_consistency']:.6f}", flush=True)
        sample = out[mode]["codes"]
        report[f"{mode}_rank_consistency"] = out[mode]["rank_consistency"]
        report[f"{mode}_codes"] = sample
    base = torch.tensor(out["U"]["codes"]) if "U" in out else None
    for mode in ("R", "N"):
        if mode in out and base is not None:
            report[f"{mode}_changed_vs_unsupervised"] = float(
                (torch.tensor(out[mode]["codes"]) != base).float().mean())
    report["note"] = ("this run exists only to measure gate 3 on a fitted model; gate 1 and 2 "
                      "come from gate_two_arm.json")
    for mode in ("R", "N", "U"):
        report.pop(f"{mode}_codes", None)
    target = root / "results" / "research" / "issue155" / "gate_post_fit.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
