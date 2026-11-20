"""Issue #223 smoke: a short run of the real arm modules with L_structure attached.

What is real here: the arm's `RQVAE`, its three-level `RQLayer` with Sinkhorn on
the last level, its `losses` registry, its `health_verdict` / `codeword_spacing`
gates, and the frozen Teacher saved by issue #222.  What is short: the loop --
a handful of steps instead of 3000 epochs, with `lambda=0` run through the same
entry as the control.

The run reports, per step: reconstruction and VQ loss, structure loss, the
gradient norm each group actually received, the parameter step each group took,
and every few steps the full SID health (L1/L2/L3 usage, max bucket, collision,
code-word spacing).  If a health gate fails the run stops and writes
`health_stop.json`; the head loss falling is never allowed to hide a collapse.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "stage2_RQ-VAE" / "TIGER_drq_D1"))

import train_rqvae as trainer  # noqa: E402  (the arm's own module)
import trainable_student as ts  # noqa: E402
from teacher_model import load_frozen_teacher  # noqa: E402

STEPS = 40
WARMUP_STEPS = 200
BATCH = 512
INSTANCES = 256
NODES = 3
SEED = 42
AUX_SHARE = 0.10
EVAL_EVERY = 10
STRUCTURE_ID = "directed"


def _sha(state_dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _health(model, embeddings, initial_spacing) -> dict:
    """The arm's own gate, plus the numbers it judged (so a raise is still recorded)."""
    with torch.no_grad():
        tokens = model.get_indices(embeddings).cpu().numpy()
    codebook = model.rq.vq_layers[0].get_code_embs()
    spacing = trainer.codeword_spacing(codebook)
    counts = np.bincount(tokens[:, 0], minlength=int(codebook.shape[0]))
    measured = {
        "spacing": spacing,
        "spacing_ratio": spacing / initial_spacing if initial_spacing > 0 else float("inf"),
        "max_bucket_l1": float(counts.max() / max(len(tokens), 1)),
        "l1_used": int((counts > 0).sum()),
        "l2_used": int(len(np.unique(tokens[:, :2], axis=0))),
        "l3_used": int(len(np.unique(tokens[:, :3], axis=0))),
        "full_collision": float(1.0 - len(np.unique(tokens[:, :3], axis=0)) / len(tokens)),
    }
    try:
        trainer.health_verdict(tokens, codebook, initial_spacing)
        measured["gate"] = "pass"
    except RuntimeError as error:
        measured["gate"] = "fail"
        measured["gate_message"] = str(error)
    return measured


def run(lambda_value: float, device, embeddings: torch.Tensor, teacher, pool, *,
        steps: int = STEPS, schedule: str = "full", rule: str = "capped") -> dict:
    torch.manual_seed(SEED)
    config = trainer._tokenizer_config()
    model = trainer.RQVAE(config, in_dim=embeddings.shape[1], distance=trainer.DISTANCE).to(device)
    head = ts.build_head(dim=config.codebook_dim, n_subtypes=47, n_orbits=376).to(device)
    if trainer.XAVIER_INIT:
        trainer.initialize_tiger_weights(model)
    # The arm's real initialisation: k-means the encoder output into every level.
    # Without it the codebooks start random and the L1 bucket gate fires on step 10
    # for a reason that has nothing to do with the structure gradient.
    model.init_codebook(embeddings.to(device))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                                  lr=trainer.LR, weight_decay=trainer.WEIGHT_DECAY)
    ts.assert_optimizer_covers(optimizer, model, head)

    generator = torch.Generator().manual_seed(SEED)
    initial_spacing = trainer.codeword_spacing(model.rq.vq_layers[0].get_code_embs())
    order = torch.randperm(len(embeddings), generator=generator)
    position = 0
    calibration: list[tuple[float, float, list[float]]] = []
    weight = 0.0 if lambda_value <= 0 else None
    log = {"lambda": lambda_value, "schedule": schedule, "total_steps": steps,
           "steps": [], "health": [], "calibration": []}
    # The real entry warms up for 50 epochs; at 49 batches an epoch that is ~2450
    # steps, so its first 200 steps run at 0.04-8% of LR.  Reproducing that shape is
    # what separates "the structure path destabilises the codebook" from "any step at
    # full LR destroys a k-means initialisation".
    warmup = max(1, trainer.WARMUP_EPOCHS * int(np.ceil(len(embeddings) / BATCH)))
    log["initial_health"] = _health(model, embeddings.to(device), initial_spacing)
    structure_batch = pool["nodes"]
    adjacency = pool["adjacency"].to(device)
    structure_items = embeddings[structure_batch.reshape(-1)].to(device)

    for step in range(1, steps + 1):
        if position + BATCH > len(order):
            order = torch.randperm(len(embeddings), generator=generator)
            position = 0
        index = order[position:position + BATCH]
        position += BATCH
        batch = embeddings[index].to(device)

        if schedule == "warmup":
            scale = min(1.0, (step + 1) / warmup)
            for group in optimizer.param_groups:
                group["lr"] = trainer.LR * scale
        optimizer.zero_grad(set_to_none=True)
        before = {name: parameter.detach().clone()
                  for name, parameter in model.named_parameters()
                  if name.startswith("encoder") or ".embed." in name}
        reconstructed, parts, _, _ = model(batch)
        main = trainer.CRITERION(reconstructed, batch, parts)[0]
        structure, info = ts.structure_loss(model, head, structure_items,
                                            instances=len(structure_batch), nodes=structure_batch.shape[1],
                                            teacher=teacher, adjacency=adjacency)
        report = ts.gradient_report(structure, model, head, teacher=teacher)
        ts.assert_learning_path(report)
        if lambda_value > 0:
            if weight is None:
                main_report = ts.gradient_report(main, model, head)
                calibration.append((main_report, report))
                log["calibration"].append({"main_encoder": main_report["encoder"],
                                           "main_codebooks": main_report["codebooks"],
                                           "aux_encoder": report["encoder"],
                                           "aux_codebooks": report["codebooks"]})
                if len(calibration) >= 5:
                    encoder_ref = float(np.median([c[0]["encoder"] for c in calibration]))
                    book_ref = [float(np.median([c[0]["codebooks"][level] for c in calibration]))
                                for level in range(len(model.rq.vq_layers))]
                    encoder_aux = float(np.median([c[1]["encoder"] for c in calibration]))
                    book_aux = [float(np.median([c[1]["codebooks"][level] for c in calibration]))
                                for level in range(len(model.rq.vq_layers))]
                    if rule == "encoder_only":
                        # The arm's original calibration: encoder share only, the
                        # codebook share is whatever falls out.  This is the arm that
                        # collapsed in #221 D2-D4 and again on the first smoke.
                        weight = float(AUX_SHARE * encoder_ref / max(encoder_aux, 1e-12))
                    else:
                        weight = ts.safe_aux_weight(encoder_reference=encoder_ref,
                                                encoder_aux=encoder_aux,
                                                codebook_reference=book_ref,
                                                    codebook_aux=book_aux,
                                                    share=AUX_SHARE)
                    log["weight_rule"] = {"encoder_reference": encoder_ref,
                                          "encoder_aux": encoder_aux,
                                          "codebook_reference": book_ref,
                                          "codebook_aux": book_aux, "weight": weight}
            total = main + (weight or 0.0) * structure
        else:
            total = main
        total.backward()
        step_norms = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if name.startswith("encoder") or ".embed." in name:
                step_norms[name] = float(parameter.grad.detach().pow(2).sum() ** 0.5)
        trainer.torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.GRADIENT_CLIP_NORM)
        optimizer.step()
        moved = {name: float((before[name] - parameter.detach()).abs().max())
                 for name, parameter in model.named_parameters() if name in before}
        log["steps"].append({
            "step": step, "main": float(main.detach()), "structure": float(structure.detach()),
            "weight": weight, "encoder_grad": report["encoder"],
            "codebook_grad": report["codebooks"], "head_grad": report["head"],
            "encoder_step": max(v for k, v in moved.items() if k.startswith("encoder")),
            "codebook_step": [max(v for k, v in moved.items() if "vq_layers.%d.embed" % level in k)
                              for level in range(len(model.rq.vq_layers))],
            "grad_norms": step_norms,
        })
        if step % EVAL_EVERY == 0 or step == steps:
            verdict = _health(model, embeddings.to(device), initial_spacing)
            log["health"].append({"step": step, **verdict})
            # The arm's own gate stops the run in place.  The absolute Stage3 entry
            # gate (0.10 collision) is NOT a training-time stop: a 512-row batch
            # coarsens the codes on its own, control included, so the comparison is
            # made against lambda=0 at the same step (see `compare`).
            if verdict["gate"] == "fail":
                log["health_stop"] = {"step": step, "message": verdict.get("gate_message"),
                                      "measured": verdict}
                break
    log["state_sha"] = _sha(model.state_dict())
    log["weight"] = weight
    return log


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embeddings = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                          "sentence_t5.npy").astype("float32"))
    teacher, teacher_record = load_frozen_teacher(HERE / "teacher" / "teacher_directed_v1.pth")
    teacher.to(device).eval()

    sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))
    from directed_role import instance_targets, load_pool, load_vocabulary
    vocab = load_vocabulary("directed")
    families = load_pool("train", vocab, max_instances=64)
    nodes, orbits, subtypes = families[STRUCTURE_ID.upper() if STRUCTURE_ID.upper() in families else "K3"]
    from directed_graphlets import load_adjacency
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    edges = set((rows.astype(np.int64) * n_items + cols).tolist())
    chosen = nodes[:INSTANCES]
    adjacency = np.zeros((len(chosen), chosen.shape[1], chosen.shape[1]), dtype=np.float32)
    for i in range(chosen.shape[1]):
        for j in range(chosen.shape[1]):
            if i == j:
                continue
            keys = chosen[:, i].astype(np.int64) * n_items + chosen[:, j]
            adjacency[:, i, j] = [int(key) in edges for key in keys]
    pool = {"nodes": torch.from_numpy(chosen.astype(np.int64)),
            "adjacency": torch.from_numpy(adjacency)}

    results = {"teacher": {"sha256": teacher_record.get("extra", {}).get("pool_sha256"),
                           "path": str(HERE / "teacher" / "teacher_directed_v1.pth")},
               "config": {"steps": STEPS, "batch": BATCH, "instances": len(chosen),
                          "nodes": chosen.shape[1], "seed": SEED, "aux_share": AUX_SHARE},
               "runs": {}}
    started = time.time()
    for schedule, steps, rule in (("warmup", WARMUP_STEPS, "capped"),
                                  ("full", STEPS, "capped"),
                                  ("encoder_only", 20, "encoder_only")):
        for lambda_value in (0.0, 1.0):
            if rule == "encoder_only" and lambda_value == 0.0:
                continue
            name = f"{schedule}_{lambda_value}"
            print(f"smoke {name}", flush=True)
            results["runs"][name] = run(lambda_value, device, embeddings, teacher, pool,
                                        steps=steps, schedule=schedule, rule=rule)
            last = results["runs"][name]["steps"][-1]
            print(json.dumps({"run": name, "last": {k: last[k] for k in
                              ("main", "structure", "encoder_grad", "codebook_grad", "weight")}}), flush=True)
    comparisons = {}
    for schedule in ("warmup", "full", "encoder_only"):
        control = results["runs"].get(f"{schedule}_0.0")
        treatment = results["runs"].get(f"{schedule}_1.0")
        if not control or not treatment:
            continue
        last_control = control["health"][-1] if control["health"] else control["initial_health"]
        last_treatment = treatment["health"][-1] if treatment["health"] else treatment["initial_health"]
        comparisons[schedule] = {
            "control_collision": last_control["full_collision"],
            "treatment_collision": last_treatment["full_collision"],
            "collision_delta": last_treatment["full_collision"] - last_control["full_collision"],
            "control_max_bucket": last_control["max_bucket_l1"],
            "treatment_max_bucket": last_treatment["max_bucket_l1"],
            "control_l3_used": last_control["l3_used"],
            "treatment_l3_used": last_treatment["l3_used"],
            "steps": [control["steps"][-1]["step"], treatment["steps"][-1]["step"]],
        }
    results["comparisons"] = comparisons
    results["wall_seconds"] = round(time.time() - started, 1)
    (HERE / "smoke_trainable_student.json").write_text(json.dumps(results, indent=1))
    stops = [run_log.get("health_stop") for run_log in results["runs"].values() if run_log.get("health_stop")]
    degradations = {name: value for name, value in comparisons.items()
                    if value["collision_delta"] > 0.05 or value["treatment_max_bucket"] > 0.25}
    if stops or degradations:
        (HERE / "health_stop.json").write_text(json.dumps(
            {"gates": stops, "treatment_vs_control": degradations}, indent=1))
    print(json.dumps({"wall": results["wall_seconds"],
                      "health_stop": bool(stops),
                      "final_health": {k: results["runs"][k]["health"][-1] for k in results["runs"]}}))


if __name__ == "__main__":
    main()
