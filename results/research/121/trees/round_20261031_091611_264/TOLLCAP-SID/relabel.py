"""TOLLCAP-SID: a capacity-frozen, behaviour-funded relabelling of the frozen SID.

Shape A of round four: the codebooks, the cell set and every cell's SIZE stay
bit-identical; only which boundary items occupy which existing cell is re-decided,
inside each level-1 group, by one optimal-transport solve whose cost is

    toll(i,s)  = d_P(phi(x_i), m_s) - d_P(phi(x_i), m_t(i))        (curved)
    Delta(i,s) = toll(i,s) - lambda * (A(i,s) - A(i,t(i)))          (behaviour)
    lambda     = eps_rec / sigma_A

with ``eps_rec`` the quantizer's own noise floor and ``A`` the successor PPMI of
the training transitions. Nothing is trained; the frozen Stage-2 checkpoint and
its SID are read-only inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
PARQUET = REPO / "results" / "stage0_build_parquet"
TEXT = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
BASELINE = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
OUTPUT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_TOLLCAP-SID"
N_ITEMS = 24587
CYCLE_CAP = 0.05


def hyperbolic(points: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    difference = (points[:, None, :] - centres[None, :, :]).pow(2).sum(dim=-1)
    p_sq = points.pow(2).sum(dim=-1)[:, None]
    c_sq = centres.pow(2).sum(dim=-1)[None, :]
    argument = 1.0 + 2.0 * difference / ((1.0 - p_sq) * (1.0 - c_sq)).clamp_min(1e-12)
    return torch.acosh(argument.clamp_min(1.0 + 1e-12))


def main() -> int:
    import train_rqvae as trainer
    from model.model import RQVAE

    text = np.load(TEXT).astype(np.float32)
    config = trainer._tokenizer_config()
    model = RQVAE(config, in_dim=text.shape[1])
    state = torch.load(BASELINE / "rqvae_best.pth", map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        latents = model.encoder(torch.from_numpy(text)).double()
    # phi acts in the QUANTIZER's input space: the codebooks live there too, so
    # the text embedding has to go through the frozen encoder first
    norm = latents.norm(dim=-1, keepdim=True)
    scale = 1.0 / float(norm.max())
    points = torch.tanh(scale * norm) * latents / norm.clamp_min(1e-12)
    codebooks = [state["rq.vq_layers.%d.embed.weight" % level].double() for level in range(3)]
    baseline = json.loads((BASELINE / "item_sids.json").read_text())
    items = sorted(int(key) for key in baseline)
    triples = np.asarray([baseline[str(item)][:3] for item in items], dtype=np.int64)
    row_of = {item: row for row, item in enumerate(items)}

    centres = torch.empty(len(items), latents.shape[1], dtype=torch.float64)
    for row, (a, b, d) in enumerate(triples):
        total = codebooks[0][a] + codebooks[1][b] + codebooks[2][d]
        length = float(total.norm())
        centres[row] = float(np.tanh(scale * length)) * total / max(length, 1e-12)
    # the noise floor, computed on a sample of items against their own cell
    sample = torch.arange(0, len(items), max(1, len(items) // 4096))
    own_distance = hyperbolic(points[sample], centres[sample]).diagonal()
    eps_rec = float(own_distance.pow(2).mean().sqrt())

    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    counts: dict[int, dict[int, float]] = {}
    incoming = np.zeros(N_ITEMS, dtype=np.float64)
    total = 0.0
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        sequence = [int(item) for item in history]
        if target is not None:
            sequence.append(int(target))
        for first, second in zip(sequence[:-1], sequence[1:]):
            counts.setdefault(first, {})
            counts[first][second] = counts[first].get(second, 0.0) + 1.0
            incoming[second] += 1.0
            total += 1.0
    eps_p = 1.0 / max(total, 1.0)
    share = incoming / max(incoming.sum(), 1e-12)

    cell_key = {tuple(triple): row for row, triple in enumerate(triples)}
    cell_share = np.zeros(len(items))
    for item in range(N_ITEMS):
        row = row_of.get(item)
        if row is not None:
            cell_share[row] += share[item]

    # A(i,s) over the OBSERVED successor cells only (the transitions the split has)
    probability: list[dict[int, float]] = []
    for row, item in enumerate(items):
        table = counts.get(item)
        if not table:
            probability.append({})
            continue
        mass = sum(table.values())
        aggregate: dict[int, float] = {}
        for target, count in table.items():
            column = row_of.get(target)
            if column is None:
                continue
            aggregate[column] = aggregate.get(column, 0.0) + count / mass
        probability.append(aggregate)
    observed = [(row, column, value) for row, table in enumerate(probability)
                for column, value in table.items()]
    values = np.asarray([np.log((value + eps_p) / (cell_share[column] + eps_p))
                         for _, column, value in observed])
    sigma_a = float(values.std())
    lam = eps_rec / max(sigma_a, 1e-12)

    def advantage(row: int, column: int) -> float:
        value = probability[row].get(column, 0.0)
        return float(np.log((value + eps_p) / (cell_share[column] + eps_p)))

    new_triples = triples.copy()
    tolls: list[float] = []
    moved_flags = np.zeros(len(items), dtype=bool)
    for level1 in np.unique(triples[:, 0]):
        members = np.flatnonzero(triples[:, 0] == level1)
        keys = triples[members][:, 1:]
        unique_keys, inverse, sizes = np.unique(keys, axis=0, return_inverse=True,
                                                return_counts=True)
        slots = np.repeat(np.arange(len(unique_keys)), sizes)
        cost = np.zeros((len(members), len(slots)), dtype=np.float64)
        base_toll = hyperbolic(points[members], centres[members]).diagonal().numpy()
        for position, slot in enumerate(slots):
            target_rows = members[inverse == slot]
            to_cells = hyperbolic(points[members], centres[target_rows]).numpy()
            toll = to_cells.min(axis=1) - base_toll
            toll = np.maximum(toll, 0.0)
            gain = np.asarray([np.mean([advantage(row, column) for column in target_rows])
                               - advantage(row, row) for row in members])
            cost[:, position] = toll - lam * gain
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            chosen = tuple(int(value) for value in unique_keys[slots[column]])
            item = members[row]
            if chosen != tuple(int(value) for value in triples[item][1:]):
                new_triples[item, 1:] = chosen
                tolls.append(float(cost[row, column]))
                moved_flags[item] = True

    moved = int(moved_flags.sum())
    cap = int(CYCLE_CAP * len(items))
    if moved > cap:
        order = np.argsort(np.asarray(tolls))[:cap]
        candidates = np.flatnonzero(moved_flags)
        revert = np.delete(candidates, order, axis=0)
        new_triples[revert] = triples[revert]
        moved = cap

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "item_sids.json").write_text(json.dumps(
        {str(item): [int(value) for value in new_triples[row]]
         for row, item in enumerate(items)}))
    diagnostics = {
        "baseline": str(BASELINE.relative_to(REPO)),
        "items": len(items),
        "moved": moved,
        "lambda": lam,
        "eps_rec": eps_rec,
        "sigma_A": sigma_a,
        "toll_median": float(np.median(tolls)) if tolls else 0.0,
        "prefix_counts": {
            "L1": int(len(np.unique(new_triples[:, 0]))),
            "L2": int(len(np.unique(new_triples[:, :2], axis=0))),
            "L3": int(len(np.unique(new_triples, axis=0)))},
        "baseline_prefix_counts": {
            "L1": int(len(np.unique(triples[:, 0]))),
            "L2": int(len(np.unique(triples[:, :2], axis=0))),
            "L3": int(len(np.unique(triples, axis=0)))},
        "SID_changed": int((new_triples != triples).any(axis=1).sum()),
        "rule": "capacity-frozen optimal transport inside each level-1 group, cost = "
                "curved toll - lambda * PPMI advantage; training split only",
    }
    (OUTPUT / "relabel_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
