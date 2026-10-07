"""Frozen diagnostic: what does the residual curvature actually control?

The assignment curvature turned out to be nearly inert on the residual side:
under the s-pin, sqrt(c) rescales the metric factor and the ball depth together,
moving behaviour distances by 0.13% (iter96). The residual curvature is the
other half of the geometry, and it is not pinned, so it is asked five questions
here, one at a time, with one residual transition moving at a time and the
codebook frozen:

  1. magnitude: does ||r_l|| move, and how much information does each level
     actually strip (E_l = ||r_{l+1}|| / ||r_l||)?
  2. direction: does cos(r_l, r_l_parent) show a rotation rather than a rescale?
  3. behaviour: is the gap between a true successor and a random item, measured
     on the residual directions, preserved? This is the gate that matters, since
     behaviour lives in direction.
  4. assignment: how far does the code assignment move, and is it ordered rather
     than random churn?
  5. recomposition: does r_3 still describe the information the codes failed to
     capture? z_hat = sum of the restored code contributions plus r_3, compared
     against the encoder latent. A residual geometry that cannot be put back
     together is not a residual, however good its other statistics look.

Nothing is trained.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
from model import RQVAE
from model.layers import (
    _hyperbolic_residual,
    _pairwise_poincare_distance_tangents,
)


def tokenizer_config(residual_curvatures: tuple[float, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_sizes=experiment.HIDDEN_SIZES,
        codebook_num=3,
        codebook_size=experiment.CODEBOOK_SIZE,
        codebook_dim=experiment.CODEBOOK_DIM,
        dropout=0.0,
        beta=experiment.BETA,
        vq_type=experiment.VQ_TYPE,
        ema_decay=experiment.EMA_DECAY,
        fix_code_embs=False,
        sk_epsilon=experiment.SK_EPSILON,
        sk_iters=experiment.SK_ITERS,
        layer_curvatures=experiment.LAYER_CURVATURES,
        layer_working_radii=experiment.LAYER_WORKING_RADII,
        pin_in_s_coordinates=experiment.PIN_IN_S_COORDINATES,
        layer_residual_curvatures=residual_curvatures,
    )


def transition_pairs(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    successors: list[int] = []
    for history, target in zip(
        train_frame["seen_history"].to_numpy(),
        train_frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) == 0:
            continue
        sources.append(int(history[-1]))
        successors.append(int(target))
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(successors, dtype=np.int64),
    )


@torch.no_grad()
def traverse(
    model: RQVAE,
    embeddings: torch.Tensor,
    rows: np.ndarray,
    batch_size: int = 4096,
) -> dict:
    """Run the stack and keep every quantity the five questions need.

    Residuals are the ones handed to the next level, after the magnitude is
    restored, because that is what the next pin consumes and therefore the thing
    a residual curvature is supposed to reshape.
    """
    device = embeddings.device
    c_quant = tuple(float(c) for c in experiment.LAYER_CURVATURES)
    layers = model.rq.vq_layers
    n = len(rows)
    n_codes = layers[0].n_embed
    tokens = np.empty((n, len(layers)), dtype=np.int64)
    margins = np.empty((n, len(layers)), dtype=np.float64)
    usage = np.zeros((len(layers), n_codes), dtype=np.float64)
    # Pinned inputs per level: the latents the assignment actually saw.
    pinned_all: list[list[torch.Tensor]] = [[] for _ in layers]
    # Restored residuals per level: r_1 is the encoder latent, r_2 and r_3 are
    # what each subtraction produced for the next level.
    residual_all: list[list[torch.Tensor]] = [[] for _ in layers]
    # Restored code contributions, summed by the trainer into the decoder input.
    quant_all: list[list[torch.Tensor]] = [[] for _ in layers]

    for start in range(0, n, batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(rows[start : start + batch_size]))
        batch = embeddings[chunk.to(device)]
        latent = model.encoder(batch)
        residual = latent
        previous_codes = None
        for level, layer in enumerate(layers):
            c2 = c_quant[level]
            c_res = float(model.rq._residual_curvature(level))
            source = residual
            target_norm = model.rq._radius_for_level(level, c2, source)
            pinned = model.rq._pin_to_radius(source, target_norm)
            distances = _pairwise_poincare_distance_tangents(
                pinned, layer.get_code_embs(), c2
            )
            two = torch.topk(distances, k=2, dim=1, largest=False).values
            margins[start : start + batch.shape[0], level] = (
                two[:, 1] - two[:, 0]
            ).cpu().numpy()
            chosen = layer._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            tokens[start : start + batch.shape[0], level] = chosen.cpu().numpy()
            usage[level] += np.bincount(
                chosen.cpu().numpy(), minlength=n_codes
            ).astype(np.float64)
            quant = layer.embed_code(chosen)
            residual = model.rq._restore_norm(
                _hyperbolic_residual(pinned, quant, c_res), source, target_norm
            )
            pinned_all[level].append(pinned)
            residual_all[level].append(residual)
            quant_all[level].append(
                model.rq._restore_norm(quant, source, target_norm)
            )
            previous_codes = chosen

    return {
        "tokens": tokens,
        "margins": margins,
        "usage": usage,
        "pinned": [torch.cat(p, dim=0) for p in pinned_all],
        "residual": [torch.cat(r, dim=0) for r in residual_all],
        "quant": [torch.cat(q, dim=0) for q in quant_all],
    }


def summarise(values: np.ndarray) -> dict:
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]),
        "mean": float(values.mean()), "std": float(values.std(ddof=1)),
    }


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(experiment.DIAGNOSTIC_SEED)
    torch.manual_seed(experiment.DIAGNOSTIC_SEED)

    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    n_items = embeddings.shape[0]
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(
        len(source_ids),
        size=min(experiment.DIAGNOSTIC_PAIRS, len(source_ids)),
        replace=False,
    )
    source_ids, successor_ids = source_ids[pick], successor_ids[pick]
    negative_ids = generator.integers(0, n_items, size=len(source_ids))
    clash = negative_ids == successor_ids
    negative_ids[clash] = (negative_ids[clash] + 1) % n_items
    # The corpus read is what drives the assignment statistics; the behaviour
    # read is a smaller sample so the cosine work stays cheap.
    corpus_rows = np.arange(n_items, dtype=np.int64)

    checkpoint = torch.load(
        experiment.PARENT_CKPT, map_location=device, weights_only=False
    )
    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "diagnostic_pairs": int(len(source_ids)),
    }

    def run(residual_curvatures):
        model = RQVAE(
            tokenizer_config(residual_curvatures), in_dim=embeddings.shape[1]
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    # ---- parent reference
    base_model = run((1.0, 1.0, 1.0))
    base_c = traverse(base_model, embeddings, corpus_rows)
    base_latent = base_c["residual"][0]
    base_norm = [
        torch.linalg.vector_norm(base_c["residual"][lv], dim=-1).cpu().numpy()
        for lv in range(3)
    ]
    lines.append("reference: parent c_res = (1, 1, 1)")
    for lv in range(3):
        s = summarise(base_norm[lv])
        lines.append(
            f"   ||r{lv + 1}|| p5/p25/p50/p75/p95="
            f"{s['p5']:.5f}/{s['p25']:.5f}/{s['p50']:.5f}/"
            f"{s['p75']:.5f}/{s['p95']:.5f}"
        )
    lines.append(
        f"   E1=||r2||/||r1||={base_norm[1].mean() / base_norm[0].mean():.5f}   "
        f"E2=||r3||/||r2||={base_norm[2].mean() / base_norm[1].mean():.5f}"
    )
    result["parent"] = {
        "residual_norm": [summarise(v) for v in base_norm],
        "E1": float(base_norm[1].mean() / base_norm[0].mean()),
        "E2": float(base_norm[2].mean() / base_norm[1].mean()),
    }

    def behaviour_gap(run_c, level):
        """Gap between a true successor and a random item on residual directions.

        Indexed against the corpus read, so the row ids are item ids.
        """
        r = run_c["residual"][level]
        unit = r / torch.linalg.vector_norm(r, dim=-1, keepdim=True).clamp_min(1e-12)
        idx_s = torch.from_numpy(source_ids).to(device)
        idx_b = torch.from_numpy(successor_ids).to(device)
        idx_x = torch.from_numpy(negative_ids).to(device)
        pos = (unit[idx_s] * unit[idx_b]).sum(-1).mean()
        neg = (unit[idx_s] * unit[idx_x]).sum(-1).mean()
        return float(pos - neg)

    def recomposition_error(model, run_c):
        """Distance between z_hat = sum of restored codes + r_3 and the latent.

        This is the check that the residual is still "what the codes missed".
        The sum is the trainer's own reconstruction rule, not a new one.
        """
        total = run_c["quant"][0] + run_c["quant"][1] + run_c["quant"][2]
        total = total + run_c["residual"][2]
        diff = total - run_c["residual"][0]
        rel = torch.linalg.vector_norm(diff, dim=-1) / torch.linalg.vector_norm(
            run_c["residual"][0], dim=-1
        ).clamp_min(1e-12)
        return float(rel.mean())

    base_gap = {lv: behaviour_gap(base_c, lv) for lv in (1, 2)}
    base_recomp = recomposition_error(base_model, base_c)
    lines.append(
        f"   behaviour gap r2={base_gap[1]:+.5f}  r3={base_gap[2]:+.5f}   "
        f"recomposition={base_recomp:.5f}"
    )
    result["parent"]["behaviour_gap"] = {f"r{lv + 2}": base_gap[lv] for lv in (1, 2)}
    result["parent"]["recomposition_error"] = base_recomp

    # ---- one transition at a time
    result["sweeps"] = {}
    for target in experiment.SWEEP_TARGETS:
        lines.append(
            f"sweep: c_res,{target + 1} (produces r_{target + 2}), "
            f"other transition at 1.0"
        )
        result["sweeps"][f"c_res_{target + 1}"] = {}
        for value in experiment.SWEEP_VALUES:
            curvatures = [1.0, 1.0, 1.0]
            curvatures[target] = value
            model = run(tuple(curvatures))
            run_c = traverse(model, embeddings, corpus_rows)
            norms = [
                torch.linalg.vector_norm(run_c["residual"][lv], dim=-1).cpu().numpy()
                for lv in range(3)
            ]
            # ``target`` is the level whose subtraction runs at this curvature.
            # That subtraction produces the next residual, so the affected
            # residual index is target+1 and the downstream one target+2.
            produced = target + 1
            entry: dict = {
                "c_res": curvatures,
                "residual_norm": [summarise(v) for v in norms],
                "E1": float(norms[1].mean() / norms[0].mean()),
                "E2": float(norms[2].mean() / norms[1].mean()),
                "cosine_to_parent": {},
                "behaviour_gap": {},
                "assignment_flip": {},
                "mean_margin": {},
            }
            for lv in range(3):
                cos = torch.nn.functional.cosine_similarity(
                    run_c["residual"][lv], base_c["residual"][lv], dim=-1
                ).mean()
                entry["cosine_to_parent"][f"r{lv + 1}"] = float(cos)
                if lv in (1, 2):
                    entry["behaviour_gap"][f"r{lv + 1}"] = behaviour_gap(run_c, lv)
                entry["assignment_flip"][f"L{lv + 1}"] = float(
                    (run_c["tokens"][:, lv] != base_c["tokens"][:, lv]).mean()
                )
                entry["mean_margin"][f"L{lv + 1}"] = float(
                    run_c["margins"][:, lv].mean()
                )
            entry["recomposition_error"] = recomposition_error(model, run_c)
            affected = f"r{produced + 1}"
            # The last residual has nothing downstream, so the downstream
            # readings are null rather than a wrapped index.
            has_downstream = produced + 1 < len(norms)
            downstream = f"r{produced + 2}" if has_downstream else affected
            entry["produced_norm_change_pct"] = 100.0 * (
                norms[produced].mean() / base_norm[produced].mean() - 1.0
            )
            entry["downstream_norm_change_pct"] = (
                100.0 * (
                    norms[produced + 1].mean() / base_norm[produced + 1].mean()
                    - 1.0
                )
                if has_downstream
                else None
            )
            entry["produced_cosine_to_parent"] = entry["cosine_to_parent"][affected]
            entry["downstream_cosine_to_parent"] = (
                entry["cosine_to_parent"][downstream] if has_downstream else None
            )
            entry["behaviour_gap_drop"] = {
                f"r{lv + 1}": entry["behaviour_gap"][f"r{lv + 1}"] - base_gap[lv]
                for lv in (1, 2)
            }
            entry["recomposition_growth"] = (
                entry["recomposition_error"] / base_recomp - 1.0
            )
            result["sweeps"][f"c_res_{target + 1}"][f"{value}"] = entry

            downstream_pct = (
                f"{entry['downstream_norm_change_pct']:+6.2f}%"
                if has_downstream
                else "    n/a"
            )
            downstream_cos = (
                f"{entry['downstream_cosine_to_parent']:.5f}"
                if has_downstream
                else "   n/a"
            )
            lines.append(
                f"   c_res,{target + 1}={value:<5} "
                f"|{affected}|={entry['produced_norm_change_pct']:+6.2f}%  "
                f"|{downstream}|={downstream_pct}  "
                f"cos({affected})={entry['produced_cosine_to_parent']:.5f}  "
                f"cos({downstream})={downstream_cos}"
            )
            lines.append(
                f"        behaviour gap r2={entry['behaviour_gap']['r2']:+.5f}"
                f" ({entry['behaviour_gap_drop']['r2']:+.5f})  "
                f"r3={entry['behaviour_gap']['r3']:+.5f}"
                f" ({entry['behaviour_gap_drop']['r3']:+.5f})  "
                f"recomposition={entry['recomposition_error']:.5f}"
                f" ({entry['recomposition_growth'] * 100:+.2f}%)  "
                f"flip L2={entry['assignment_flip']['L2'] * 100:5.2f}% "
                f"L3={entry['assignment_flip']['L3'] * 100:5.2f}%"
            )
            del model, run_c
            torch.cuda.empty_cache()

    # ---- screen against the thresholds
    lines.append("screen")
    survivors: list[str] = []
    for key, block in result["sweeps"].items():
        for value, entry in block.items():
            if value == "1.0":
                continue
            geometry = (
                abs(entry["produced_norm_change_pct"]) > 1.0
                or (
                    has_downstream
                    and (
                        abs(entry["downstream_norm_change_pct"]) > 1.0
                        or abs(entry["downstream_cosine_to_parent"] - 1.0) > 0.01
                    )
                )
            )
            behaviour = min(entry["behaviour_gap_drop"].values()) >= -experiment.BEHAVIOUR_MAX_DROP
            recomposition = entry["recomposition_growth"] <= experiment.RECOMPOSITION_MAX_GROWTH
            churn = max(entry["assignment_flip"].values()) <= experiment.MAX_ASSIGNMENT_FLIP
            passed = geometry and behaviour and recomposition and churn
            entry["screen"] = {
                "geometry_responds": bool(geometry),
                "behaviour_preserved": bool(behaviour),
                "recomposition_ok": bool(recomposition),
                "no_churn": bool(churn),
                "passes": bool(passed),
            }
            if passed:
                survivors.append(f"{key}={value}")
            lines.append(
                f"   {key}={value:<5} geometry={int(geometry)} "
                f"behaviour={int(behaviour)} recomposition={int(recomposition)} "
                f"no_churn={int(churn)} -> {'TRAIN' if passed else 'drop'}"
            )
    result["survivors"] = survivors
    lines.append(
        f"survivors for stage 3: {survivors if survivors else 'none'}"
    )

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.DIAGNOSTIC_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.DIAGNOSTIC_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
