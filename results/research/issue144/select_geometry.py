"""Issue #144: can AF3 hard-SID structure choose a per-layer geometry externally?

The study is evaluation only: it reuses the five completed #139 geometry arms
and the frozen Euclidean baseline, recomputes every SID's structural distortion
with ONE frozen implementation on TWO independent supports (a selection support
and an untouched evaluation support), and asks whether the selector is stable.

    python3 results/research/issue144/select_geometry.py

No model is trained, no SID is regenerated, no AF3 value enters a training path.
A candidate whose SID or checkpoint is missing, or whose item space differs from
the frozen baseline, is reported as DATA_UNAVAILABLE / INCOMPARABLE instead of
being silently dropped.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULT = REPO / "results" / "stage2_RQ-VAE"
TREE = REPO / "stage2_RQ-VAE"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _structural_module():
    path = REPO / "results" / "research" / "issue143" / "structural_report.py"
    spec = importlib.util.spec_from_file_location("issue143_structural_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Each candidate's per-layer geometry, read from its own frozen constants.
CANDIDATES = [
    {"arm": "TIGER_RQ-VAE", "readout": "euclidean",
     "profile": {"L1": "E32", "L2": "E32", "L3": "E32", "curvature": None,
                 "alpha": "none"}},
    {"arm": "TIGER_mixed_rq_F0", "readout": "mixed",
     "profile": {"L1": "E16xH16", "L2": "E16xH16", "L3": "E16xH16",
                 "curvature": 1.0, "alpha": "fixed 1/2"}},
    {"arm": "TIGER_mixed_rq_G", "readout": "mixed",
     "profile": {"L1": "E16xH16", "L2": "E16xH16", "L3": "E16xH16",
                 "curvature": 1.0, "alpha": "global alpha_l"}},
    {"arm": "TIGER_mixed_rq_L", "readout": "mixed",
     "profile": {"L1": "E16xH16", "L2": "E16xH16", "L3": "E16xH16",
                 "curvature": 1.0, "alpha": "prefix alpha_l(p)"}},
    {"arm": "TIGER_mixed_rq_E", "readout": "euclidean",
     "profile": {"L1": "E16xE16", "L2": "E16xE16", "L3": "E16xE16",
                 "curvature": 0.0, "alpha": "local outer loop"}},
    {"arm": "TIGER_mixed_rq_A", "readout": "mixed",
     "profile": {"L1": "E16xH16", "L2": "E16xH16", "L3": "E16xH16",
                 "curvature": 1.0, "alpha": "fixed 1/2 + AF3 auxiliary"}},
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def readout(engine, candidate: dict, tokens, codebooks):
    import torch
    from model.mixed_rq import quantized_readout
    if candidate["readout"] == "euclidean":
        return sum(book[torch.as_tensor(tokens[:, level])]
                   for level, book in enumerate(codebooks))
    codes_e = [book[:, :16][torch.as_tensor(tokens[:, level])]
               for level, book in enumerate(codebooks)]
    codes_h = [book[:, 16:][torch.as_tensor(tokens[:, level])]
               for level, book in enumerate(codebooks)]
    return quantized_readout(codes_e, codes_h, curvature=1.0)


def measure(engine, candidate: dict, support, graph, kappa, embeddings) -> dict:
    import torch
    arm = candidate["arm"]
    sid_path = RESULT / arm / "item_sids.json"
    checkpoint = RESULT / arm / "rqvae_best.pth"
    if not sid_path.is_file() or not checkpoint.is_file():
        return {"available": False, "reason": "SID or checkpoint missing"}
    payload = json.loads(sid_path.read_text())
    tokens = engine.np.asarray([payload[str(index)] for index in range(len(payload))],
                               dtype=engine.np.int64)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    codebooks = [state[key] for key in sorted(state)
                 if key.startswith("rq.") and key.endswith("embed.weight")]
    if len(tokens) != len(embeddings):
        return {"available": False,
                "reason": "item space %d differs from the frozen %d"
                          % (len(tokens), len(embeddings))}
    with torch.no_grad():
        phi = readout(engine, candidate, tokens, codebooks)
    calibration = engine.structure.calibrate(phi, support, device=phi.device)
    result = engine.structure.structural_distortion(
        phi, support, kappa, tau=calibration["tau"], intercept=calibration["intercept"],
        n_nodes=graph["n_nodes"])
    retention = engine.neighbour_retention(phi, support["pairs"],
                                           n_nodes=graph["n_nodes"], k=16, seed=42)
    probability = torch.as_tensor(result["probability"])
    is_edge = engine.np.asarray(support["is_edge"], dtype=bool)
    brier = float(torch.mean((probability - torch.as_tensor(is_edge, dtype=torch.float32))
                             ** 2))
    codes = tokens[:, :len(codebooks)]
    full = engine.collision(codes)
    reconstruction = engine.decoder_reconstruction(state, phi, embeddings)
    return {"available": True, "distortion": float(result["distortion"]),
            "retention": float(retention["retention"]),
            "brier": brier, "usage": [int(len(engine.np.unique(codes[:, level])))
                                      for level in range(len(codebooks))],
            "collision_full": full["share"], "max_triple_bucket": full["max_bucket"],
            "reconstruction_mse": reconstruction["mse"],
            "sid_sha256": digest(sid_path), "checkpoint_sha256": digest(checkpoint)}


def pareto(rows: dict) -> list[str]:
    """Non-dominated candidates: lower distortion/brier/collision/MSE, higher retention/usage."""
    def dominates(a: dict, b: dict) -> bool:
        better_or_equal = (a["distortion"] <= b["distortion"]
                           and a["brier"] <= b["brier"]
                           and a["collision_full"] <= b["collision_full"]
                           and a["reconstruction_mse"] <= b["reconstruction_mse"]
                           and a["retention"] >= b["retention"]
                           and min(a["usage"]) >= min(b["usage"]))
        strictly = (a["distortion"] < b["distortion"]
                    or a["brier"] < b["brier"]
                    or a["collision_full"] < b["collision_full"]
                    or a["reconstruction_mse"] < b["reconstruction_mse"]
                    or a["retention"] > b["retention"]
                    or min(a["usage"]) > min(b["usage"]))
        return better_or_equal and strictly
    names = [name for name, body in rows.items() if body.get("available")]
    return sorted(name for name in names
                  if not any(dominates(rows[other], rows[name])
                             for other in names if other != name))


def spearman(pairs: list[tuple[float, float]]) -> float:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda index: values[index])
        rank = [0.0] * len(values)
        for position, index in enumerate(order):
            rank[index] = float(position)
        return rank
    left = ranks([pair[0] for pair in pairs])
    right = ranks([pair[1] for pair in pairs])
    n = len(pairs)
    mean_left = sum(left) / n
    mean_right = sum(right) / n
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = ((sum((a - mean_left) ** 2 for a in left)
                    * sum((b - mean_right) ** 2 for b in right)) ** 0.5)
    return numerator / denominator if denominator else float("nan")


def main() -> int:
    engine = _structural_module()
    graph, kappa, selection, embeddings = engine.frozen_graph()
    evaluation = engine.structure.build_support(
        graph, kappa, embeddings[graph["item_ids"]], k=16, seed=7)
    inventory, selection_rows, evaluation_rows = {}, {}, {}
    for candidate in CANDIDATES:
        arm = candidate["arm"]
        tree = TREE / arm
        spec = tree / "math_spec.json"
        inventory[arm] = {
            "profile": candidate["profile"],
            "readout": candidate["readout"],
            "math_spec_sha256": digest(spec) if spec.is_file() else None,
            "sid": str(RESULT / arm / "item_sids.json"),
            "checkpoint": str(RESULT / arm / "rqvae_best.pth"),
            "has_sid": (RESULT / arm / "item_sids.json").is_file(),
            "has_checkpoint": (RESULT / arm / "rqvae_best.pth").is_file(),
        }
        selection_rows[arm] = measure(engine, candidate, selection, graph, kappa, embeddings)
        evaluation_rows[arm] = measure(engine, candidate, evaluation, graph, kappa, embeddings)
    available = [arm for arm, body in evaluation_rows.items() if body.get("available")]
    stability = {}
    if len(available) >= 3:
        for metric in ("distortion", "retention", "brier"):
            pairs = [(selection_rows[arm][metric], evaluation_rows[arm][metric])
                     for arm in available]
            stability[metric] = spearman(pairs)
    evaluation_pareto = pareto(evaluation_rows)
    selection_pareto = pareto(selection_rows)
    payload = {
        "supports": {
            "selection": {"seed": 42, "pairs": int(len(selection["pairs"])),
                          "edges": int(selection["edges"])},
            "evaluation": {"seed": 7, "pairs": int(len(evaluation["pairs"])),
                           "edges": int(evaluation["edges"]),
                           "built_after_the_selection_support_was_frozen": True},
        },
        "inventory": inventory,
        "selection": selection_rows,
        "evaluation": evaluation_rows,
        "selection_pareto": selection_pareto,
        "evaluation_pareto": evaluation_pareto,
        "rank_stability": stability,
        "verdict": {
            "selector_stable": (bool(stability) and all(value > 0.5 for value in stability.values())
                                and selection_pareto == evaluation_pareto),
            "note": "stability needs the same ordering and the same Pareto set on the "
                    "untouched support; a single support cannot prove it",
        },
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "geometry_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"supports": payload["supports"],
                      "selection_pareto": selection_pareto,
                      "evaluation_pareto": evaluation_pareto,
                      "rank_stability": stability,
                      "verdict": payload["verdict"]}, indent=2))
    header = ("arm".ljust(24) + "D_AF3".rjust(9) + "ret@16".rjust(9)
              + "brier".rjust(9) + "min_use".rjust(9) + "coll".rjust(9) + "mse".rjust(11))
    print(header)
    for arm in available:
        row = evaluation_rows[arm]
        print(arm.ljust(24)
              + ("%.4f" % row["distortion"]).rjust(9)
              + ("%.4f" % row["retention"]).rjust(9)
              + ("%.4f" % row["brier"]).rjust(9)
              + str(min(row["usage"])).rjust(9)
              + ("%.4f" % row["collision_full"]).rjust(9)
              + ("%.2e" % row["reconstruction_mse"]).rjust(11))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
