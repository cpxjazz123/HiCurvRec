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

from model import rank_objective as ro
from model import structure_runtime as sr  # noqa: E402
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

    from model.structure_runtime import StructureSupervision
    supervision = StructureSupervision.from_repo(root, 0.48247611144034613)
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
        distance = ro.edge_distance(prefixes[-1][0].detach(), prefixes[-1][1].detach(),
                                    eval_edges_t, 1.0).numpy()
        return float(spearmanr(eval_u_t, distance).correlation)

    def codes(model) -> torch.Tensor:
        model.eval()
        model(rows[torch.from_numpy(eval_ids)])
        return model.rq.last_codes[-1].clone()

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
                terms = supervision.terms(model, batch, 1.0)
                if terms is not None:
                    loss = loss + terms
            loss.backward()
            optimizer.step()
        return {"rank_consistency": rank_consistency(model), "codes": codes(model).tolist()}

    report = {"steps": STEPS, "batch": SIZE, "eval_rows": EVAL, "d_ref": d_ref,
              "eval_edges": int(eval_edges_t.shape[0])}
    out = {}
    for mode in ("U", "R", "N"):
        out[mode] = fit(mode)
        print(f"[arm {mode}] rank_consistency={out[mode]['rank_consistency']:.6f}", flush=True)
        sample = out[mode]["codes"]
        report[f"{mode}_rank_consistency"] = out[mode]["rank_consistency"]
        report[f"{mode}_codes"] = sample
    base = torch.tensor(out["U"]["codes"])
    for mode in ("R", "N"):
        report[f"{mode}_changed_vs_unsupervised"] = float(
            (torch.tensor(out[mode]["codes"]) != base).float().mean())
    report["R_improvement"] = abs(report["R_rank_consistency"]) - abs(report["U_rank_consistency"])
    report["N_improvement"] = abs(report["N_rank_consistency"]) - abs(report["U_rank_consistency"])
    report["ratio_R_over_N"] = (report["R_improvement"] / report["N_improvement"]
                                if report["N_improvement"] not in (0.0,) else None)
    report["gate1_pass"] = bool(
        report["R_improvement"] > 0 and report["ratio_R_over_N"] is not None
        and report["ratio_R_over_N"] >= 2.0)
    report["gate2_non_inert"] = bool(report["R_changed_vs_unsupervised"] >= 0.10)
    for mode in ("R", "N"):
        report.pop(f"{mode}_codes")
    target = root / "results" / "research" / "issue155" / "gate_two_arm.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
