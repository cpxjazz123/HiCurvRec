"""Pre-flight gate for the repaired geometry arms (CLAUDE.md 6, run inline).

Run from the tree root with no arguments; reads the hardcoded arm table and
reports on whichever arm EXPERIMENT_ARM selects. Prints JSON, writes nothing,
exits non-zero on FAIL.

Addressed here after the audit:

  * the model is put in ``train()`` mode. An earlier version measured in
    ``eval()`` mode, where ``VQLayer._indices`` takes the argmin branch because
    ``self.training and self.use_sk`` is false, so level 3 was traced with a
    *different quantizer* than the one training actually uses and the calibrated
    temperature did not match the run's.
  * the fixed ball normalizer is built exactly the way the trainer builds it.
  * every level's gradient norm is reported separately, because the whole point
    of the Exp2 repair is that levels 2 and 3 must stop being inert.

Checks: ball image inside the radius and spread over it; distance finiteness,
symmetry and bounded self-distance; logit spread and softmax saturation after
calibration; per-level encoder gradient norms with all levels live; the total
objective finite with a real grad_fn; and that quantization, reconstruction and
SID export are unaffected by the trace repair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import train_rqvae as T  # noqa: E402
from hyperbolic import ball_radius, encode_ball, poincare_distance  # noqa: E402
from model import RQVAE  # noqa: E402


def check_main_names() -> list[str]:
    """Catch a stale module-level name inside ``main`` before the run starts.

    The pre-flight imports this module and exercises the loss helpers, but never
    enters ``main()``, so a reference to a constant that was deleted during
    refactoring stays invisible and only surfaces as a NameError when torchrun
    launches. That happened once with ``TANGENT_SCALE``. Walking the AST of
    ``main`` for unbound Load names turns that class of failure into a gate.
    """
    import ast
    import builtins

    tree = ast.parse((SOURCE_DIR / "train_rqvae.py").read_text())
    defined: set[str] = set(dir(builtins))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(
                t.id for t in node.targets if isinstance(t, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
    for node in ast.walk(tree):
        defined.add(getattr(node, "name", ""))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.update(a.arg for a in node.args.args)
            defined.update(a.arg for a in getattr(node.args, "kwonlyargs", []))
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)
    main_fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"),
        None,
    )
    if main_fn is None:
        return ["main() not found"]
    assigned = {
        t.id
        for n in ast.walk(main_fn)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assigned |= {
        n.target.id
        for n in ast.walk(main_fn)
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    assigned |= {
        n.id
        for n in ast.walk(main_fn)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    assigned |= {
        t.id
        for n in ast.walk(main_fn)
        if isinstance(n, (ast.For, ast.AsyncFor))
        for t in ast.walk(n.target)
        if isinstance(t, ast.Name)
    }
    assigned |= {
        n.name for n in ast.walk(main_fn) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    missing = sorted(
        {
            n.id
            for n in ast.walk(main_fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        - defined
        - assigned
    )
    return missing


def main() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report: dict = {
        "arm": T.EXPERIMENT_ARM,
        "geometry": T.GEOMETRY,
        "group_distance": T.GROUP_DISTANCE,
        "use_ball": T.USE_BALL,
        "curvature": T.CURVATURE,
        "target_radius_fraction": T.TARGET_RADIUS_FRACTION,
        "training_mode_used_for_checks": "train()",
        "device": str(device),
    }
    failures: list[str] = []

    unresolved = check_main_names()
    report["unresolved_names_in_main"] = unresolved
    if unresolved:
        failures.append(f"main() references undefined names: {unresolved}")

    T.set_seed(T.SEED)
    embeddings = T.load_embeddings(T.EMBEDDING_FILE)
    frame = pd.read_parquet(T.TRAIN_FILE, columns=["seen_history", "target"])
    ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    all_emb = torch.from_numpy(embeddings)
    train_emb = all_emb[torch.from_numpy(ids)]
    pair_dataset = T.TransitionPairs(frame)

    model = RQVAE(T._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    T.initialize_tiger_weights(model)
    # train() to match the run: the level-3 Sinkhorn branch is only taken when
    # self.training is true, so eval() would trace a different quantizer.
    model.train()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        model.init_codebook(train_emb.to(device))

    normalizer_scalar = T.resolve_normalizer(model, train_emb.to(device))
    normalizer = torch.as_tensor(normalizer_scalar, dtype=torch.float32, device=device)
    report["ball_normalizer"] = normalizer_scalar

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    devices = {str(p.device) for _, p in params}
    report["parameter_devices"] = sorted(devices)
    if len(devices) != 1 or str(device) not in devices:
        failures.append(f"parameters span devices {sorted(devices)}")

    batch = T.PAIR_BATCH_SIZE_PER_RANK
    pairs = pair_dataset.pairs[:batch]
    s_ids, t_ids = pairs[:, 0].to(device), pairs[:, 1].to(device)
    pair_batch = all_emb[torch.cat((s_ids, t_ids)).cpu()].to(device)

    # --- Exp1: the map must be a pure function of the latent ----------------
    if T.USE_BALL:
        with torch.no_grad():
            encoded = model.encoder(train_emb.to(device))
            median_norm = float(torch.median(torch.linalg.vector_norm(encoded, dim=-1)))
            z_all = encode_ball(encoded, T.CURVATURE, normalizer)
            ratio = torch.linalg.vector_norm(z_all, dim=-1) / ball_radius(T.CURVATURE)
            # Probe along a unit direction scaled to the median norm, so the
            # probe's own norm is exactly median_norm. A constant vector across
            # all coordinates would have norm median_norm * sqrt(dim).
            unit = torch.zeros(1, encoded.shape[1], device=device)
            unit[0, 0] = median_norm
            report["median_radius_fraction"] = float(
                torch.linalg.vector_norm(
                    encode_ball(unit, T.CURVATURE, normalizer), dim=-1
                ) / ball_radius(T.CURVATURE)
            )
            report["radius_fraction_percentiles"] = {
                q: float(ratio.quantile(q)) for q in (0.10, 0.50, 0.90, 0.99)
            }
            report["radius_fraction_std"] = float(ratio.std())
            report["saturated_fraction"] = float((ratio > 0.99).float().mean())
            if not bool((ratio < 1.0).all()):
                failures.append("ball image reaches or exceeds the radius")
            if report["radius_fraction_std"] <= 0.0:
                failures.append("radial coordinate is constant; hierarchy is not representable")
            # identical item, different batch companions -> identical coordinate
            probe = train_emb[0:1].to(device)
            alone = encode_ball(probe, T.CURVATURE, normalizer)[0]
            refs = []
            for seed in (0, 1, 2):
                g = torch.Generator(device="cpu").manual_seed(seed)
                other = train_emb[torch.randint(0, len(train_emb), (1024,), generator=g)]
                combined = torch.cat([probe.cpu(), other]).to(device)
                refs.append(encode_ball(combined, T.CURVATURE, normalizer)[0])
            deviation = max(float((r - alone).abs().max()) for r in refs)
            report["max_coordinate_deviation_across_batches"] = deviation
            # The map is a pure function of the latent, but the encoder matmul
            # reduces over a differently shaped batch, so float32 reassociation
            # leaves ~1e-8. Anything at that scale is rounding, not dependence.
            if deviation > 1e-6:
                failures.append(f"coordinate depends on batch membership: {deviation:.3e}")

    # --- Exp3: distance axioms on the mapped points -------------------------
    with torch.no_grad():
        residuals = T._residuals_with_trace(model, pair_batch)
        report["residual_shape"] = list(residuals.shape)
        src, cand = residuals[0, :batch], residuals[0, batch:]
        distances = T._geodesic_block(src, cand, normalizer)
        report["distance_finite"] = bool(torch.isfinite(distances).all())
        report["distance_range"] = [float(distances.min()), float(distances.max())]
        if not report["distance_finite"]:
            failures.append("distance matrix is not finite")
        if T.USE_BALL and T.GEOMETRY == "poincare":
            z = encode_ball(torch.cat((src, cand), 0), T.CURVATURE, normalizer)
            zs, zc = z[:batch], z[batch:]
            self_d = poincare_distance(zs, zs, T.CURVATURE)
            report["max_self_distance"] = float(self_d.max())
            if float(self_d.max()) > 1e-2:
                failures.append(f"self-distance too large: {float(self_d.max()):.3e}")
            asym = float(
                (poincare_distance(zs[:64], zc[:64], T.CURVATURE)
                 - poincare_distance(zc[:64], zs[:64], T.CURVATURE)).abs().max()
            )
            report["max_asymmetry"] = asym
            if asym != 0.0:
                failures.append("distance is asymmetric")
        rms = distances.detach().square().mean().sqrt().clamp_min(1e-8)
        temperature = T.calibrate_temperature(
            model, pair_batch, s_ids, t_ids, normalizer
        )
        report["resolved_temperature"] = temperature
        logits = -(distances / rms) / temperature
        spread = float((logits.max(-1).values - logits.min(-1).values).mean())
        probs = torch.softmax(logits, -1)
        own = probs.diagonal()
        saturated = float(((own < 1e-6) | (own > 1 - 1e-6)).float().mean())
        report["logit_row_spread"] = spread
        report["softmax_saturated_fraction"] = saturated
        if saturated > 0.05:
            failures.append(f"softmax saturation {saturated:.3f} exceeds 5%")

    # --- Exp2: every level must reach the encoder ---------------------------
    encoder_params = [p for n, p in params if n.startswith("encoder.")]
    per_level = {}
    for level in range(residuals.shape[0]):
        model.zero_grad(set_to_none=True)
        fresh = T._residuals_with_trace(model, pair_batch)
        loss = torch.cdist(fresh[level, :batch], fresh[level, batch:], p=2).mean()
        layer_grads = torch.autograd.grad(
            loss, encoder_params, retain_graph=False, allow_unused=True
        )
        total = sum(float(g.abs().sum()) for g in layer_grads if g is not None)
        nonzero = sum(
            1 for g in layer_grads if g is not None and float(g.abs().sum()) > 0
        )
        per_level[f"L{level + 1}"] = {
            "encoder_grad_l1": total,
            "nonzero_encoder_params": nonzero,
        }
        # What "correct" means depends on the arm. The carrying trace is the
        # repair: every level must reach the encoder. The detached trace is
        # iter48's original semantics, where each level quantizes the detached
        # error of the level above, so only level 1 can reach the encoder -- and
        # that is asserted too, so an accidental change of behaviour is caught
        # rather than silently accepted.
        if T.TRACE_GRADIENT == "carrying" and total <= 0.0:
            failures.append(
                f"carrying trace: level {level + 1} delivers no encoder gradient"
            )
        if T.TRACE_GRADIENT == "detached" and level > 0 and total > 0.0:
            failures.append(
                f"detached trace: level {level + 1} unexpectedly reaches the encoder"
            )
    report["per_level_encoder_gradient"] = per_level
    if T.TRACE_GRADIENT == "detached":
        if per_level["L1"]["encoder_grad_l1"] <= 0.0:
            failures.append("detached trace: level 1 must still reach the encoder")
        report["expected_live_levels"] = ["L1"]

    # --- the trace repair must not touch the quantizer ----------------------
    with torch.no_grad():
        recon, quant_loss, _, tokens = model(pair_batch)
        tokens_direct = model.rq(model.encoder(pair_batch))[3]
    report["trace_does_not_change_tokens"] = bool(torch.equal(tokens, tokens_direct))
    report["quantizer_reconstruction_finite"] = bool(torch.isfinite(recon).all())
    if not report["trace_does_not_change_tokens"]:
        failures.append("trace repair changed the tokens; quantization was altered")

    # --- finite gradients on a batch that contains repeat purchases --------
    # A batch where some pair has source == target puts two coincident points in
    # the distance matrix. The Euclidean sqrt used to return an infinite
    # gradient there and poison the encoder on the first backward step, which
    # only showed up later as a level-3 Sinkhorn failure. Check it directly.
    model.zero_grad(set_to_none=True)
    fresh = T._residuals_with_trace(model, pair_batch)
    probe = T.behavior_contrastive_loss(
        fresh, s_ids, t_ids, normalizer, report["resolved_temperature"]
    )
    probe.backward()
    bad = [
        n for n, q in model.named_parameters()
        if q.grad is not None and not bool(torch.isfinite(q.grad).all())
    ]
    report["params_with_nonfinite_grad"] = bad
    if bad:
        failures.append(f"non-finite gradient on {len(bad)} parameters: {bad[:5]}")
    repeat_rate = float((s_ids == t_ids).to(torch.float32).mean())
    report["repeat_purchase_fraction"] = repeat_rate
    model.zero_grad(set_to_none=True)

    # --- total objective ----------------------------------------------------
    item = train_emb[torch.arange(T.BATCH_SIZE_PER_RANK, dtype=torch.long)].to(device)
    rec, ql, _, _ = model(item)
    base, _ = model.compute_loss(item, rec, ql)
    contrastive = T.behavior_contrastive_loss(
        residuals, s_ids, t_ids, normalizer, report["resolved_temperature"]
    )
    total = base + T.BEHAVIOR_WEIGHT_MAX * contrastive
    report["contrastive_loss"] = float(contrastive.detach())
    report["total_loss"] = float(total.detach())
    report["total_grad_fn"] = type(total.grad_fn).__name__ if total.grad_fn else None
    report["total_finite"] = bool(torch.isfinite(total))
    total_grads = torch.autograd.grad(
        total, [p for _, p in params], retain_graph=True, allow_unused=True
    )
    total_nonzero = sum(
        1 for g in total_grads if g is not None and float(g.abs().sum()) > 0
    )
    report["total_nonzero_params"] = total_nonzero
    if total_nonzero == 0:
        failures.append("total objective produced no gradient")
    if not report["total_finite"]:
        failures.append("total objective is not finite")

    # --- Exp2: the fork terms must be live and must not be a no-op ---------
    out_degree = T.behaviour_out_degree(frame, len(embeddings))
    log_degree = np.log1p(out_degree.astype(np.float64))
    rank = np.argsort(np.argsort(log_degree)) / max(len(log_degree) - 1, 1)
    radius_target = torch.from_numpy(
        (T.RHO_HI - (T.RHO_HI - T.RHO_LO) * rank).astype(np.float32)
    ).to(device)
    siblings = torch.from_numpy(
        T.behaviour_siblings(frame, len(embeddings), T.FORK_SIBLINGS_PER_ITEM, T.SEED)
    ).to(device)
    item_batch = train_emb[torch.arange(T.BATCH_SIZE_PER_RANK, dtype=torch.long)].to(device)
    item_rows = torch.arange(T.BATCH_SIZE_PER_RANK, dtype=torch.long, device=device)
    pf_rng = torch.Generator(device=device); pf_rng.manual_seed(T.SEED)
    radial_term, angular_term, prefix_term, fork_stats = T.fork_geometry_terms(
        model, item_batch, item_rows, radius_target, siblings, normalizer,
        all_emb.to(device), pf_rng,
    )
    report["prefix_term"] = float(prefix_term.detach())
    report["fork_stats"] = fork_stats
    report["fork_radial_term"] = float(radial_term.detach())
    report["fork_angular_term"] = float(angular_term.detach())
    report["fork_radius_target_std"] = float(radius_target.std())
    if not torch.isfinite(radial_term) or not torch.isfinite(angular_term):
        failures.append("fork terms are not finite")
    if fork_stats["pairs_used"] == 0:
        failures.append("no behaviour-sibling pairs found; the term would be dead")
    if fork_stats["radius_std"] <= 1e-6:
        failures.append("radius has no spread; the radial term cannot order anything")
    fork_loss = T.FORK_RADIAL_WEIGHT * radial_term + T.FORK_ANGULAR_WEIGHT * angular_term
    fork_grads = torch.autograd.grad(
        fork_loss, [q for q in model.parameters() if q.requires_grad],
        retain_graph=True, allow_unused=True,
    )
    fork_nonzero = sum(
        1 for g in fork_grads if g is not None and float(g.abs().sum()) > 0
    )
    report["fork_grad_nonzero_params"] = fork_nonzero
    if fork_nonzero == 0:
        failures.append("fork terms produced no gradient")
    # the terms must actually move when the coordinates move, else they are inert
    probe = item_batch.clone().requires_grad_(True)
    pf_rng2 = torch.Generator(device=device); pf_rng2.manual_seed(T.SEED)
    r2, a2, p2, _ = T.fork_geometry_terms(
        model, probe, item_rows, radius_target, siblings, normalizer,
        all_emb.to(device), pf_rng2,
    )
    (T.FORK_RADIAL_WEIGHT * r2 + T.FORK_ANGULAR_WEIGHT * a2).backward()
    report["fork_grad_on_input"] = float(probe.grad.abs().sum()) if probe.grad is not None else 0.0
    if report["fork_grad_on_input"] <= 0.0:
        failures.append("fork terms do not depend on the representation")
    model.zero_grad(set_to_none=True)

    # --- the codebook-prefix term must reach the QUANTIZER -----------------
    if T.PREFIX_MODE != "off":
        if not torch.isfinite(prefix_term):
            failures.append("prefix term is not finite")
        grads = torch.autograd.grad(
            prefix_term, [p for p in model.parameters() if p.requires_grad],
            retain_graph=True, allow_unused=True,
        )
        names = [n for n, _ in model.named_parameters() if _.requires_grad]
        touched = [
            n for n, g in zip(names, grads)
            if g is not None and float(g.abs().sum()) > 0
        ]
        report["prefix_grad_params"] = touched[:6]
        report["prefix_touches_l1_codebook"] = any(
            "rq.vq_layers.0" in n for n in touched
        )
        if not report["prefix_touches_l1_codebook"]:
            failures.append("prefix term does not reach the L1 codebook")
        model.zero_grad(set_to_none=True)

    report["failures"] = failures
    report["status"] = "FAIL" if failures else "PASS"
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
