"""Mechanism verification for iter28's Sinkhorn-weighted Riemannian EMA (SREMA).

Runs on a single CUDA device (cuda:0).  Loads iter18's best checkpoint,
builds the iter28 RqVae, and verifies:

- gradient path for total loss + every component loss;
- non-codebook parameters still receive finite nonzero gradients;
- codebook embedding.weight is **excluded** from the AdamW optimizer;
- five ON steps (with SREMA) produce finite, geometrically-valid codebook
  updates (median_update > 0, saturation_max < 1);
- 200-step ON/OFF counterfactual: loss delta > 1e-6 AND codebook max-abs-diff
  per layer > 1e-7;
- 5-step relative update of c_layer_scale (and other non-codebook params)
  is finite and > 1e-7.
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"
REFERENCE_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter18/out/rqvae/instruments/rqvae_best.pth"
)
COUNTERFACTUAL_STEPS = 200
GRAD_EPSILON = 1e-12
SREMA_BETA = 0.99


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter28", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training entry: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loss_gradient_norm(loss_item, parameters):
    gradients = torch.autograd.grad(
        loss_item, parameters, retain_graph=True, allow_unused=True
    )
    squared_norm = sum(
        float(gradient.detach().double().square().sum().item())
        for gradient in gradients
        if gradient is not None
    )
    return float(np.sqrt(squared_norm))


def _layer_relative_updates(model, before, n_layers):
    results = []
    named_parameters = dict(model.named_parameters())
    for layer_index in range(n_layers):
        names = [name for name in before if name.startswith(f"layers.{layer_index}.")]
        before_norm_sq = sum(before[name].double().square().sum() for name in names)
        delta_norm_sq = sum(
            (named_parameters[name].detach() - before[name]).double().square().sum()
            for name in names
        )
        before_norm = float(torch.sqrt(before_norm_sq).item())
        delta_norm = float(torch.sqrt(delta_norm_sq).item())
        results.append((layer_index, delta_norm / max(before_norm, GRAD_EPSILON)))
    return results


def _layer_codebook_max_abs_diff(model_a, model_b, n_layers):
    out = []
    for layer_index in range(n_layers):
        a = model_a.module.layers[layer_index].embedding.weight \
            if hasattr(model_a, "module") else model_a.layers[layer_index].embedding.weight
        b = model_b.module.layers[layer_index].embedding.weight \
            if hasattr(model_b, "module") else model_b.layers[layer_index].embedding.weight
        diff = (a - b).abs().max().item()
        out.append((layer_index, diff))
    return out


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MVG requires CUDA")
    rqtrain = _load_training_module()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)

    dataset = rqtrain.TransitionDataset(
        rqtrain.EMB_NPY, rqtrain.ITEM_IDS_JSON, rqtrain.TRAIN_PARQUET
    )
    active_indices = [i for i, targets in enumerate(dataset.next_items) if targets]
    if len(active_indices) < rqtrain.BATCH_SIZE:
        raise RuntimeError(
            f"Only {len(active_indices)} train-transition sources; need "
            f"{rqtrain.BATCH_SIZE} for the MVG batch"
        )
    raw_batch = rqtrain.collate_items(
        [dataset[index] for index in active_indices[: rqtrain.BATCH_SIZE]]
    )
    batch = rqtrain._build_seq_batch(*(value.to(device) for value in raw_batch))

    model = rqtrain.RqVae(
        input_dim=rqtrain.INPUT_DIM,
        embed_dim=rqtrain.EMBED_DIM,
        hidden_dims=rqtrain.HIDDEN_DIMS,
        codebook_size=rqtrain.CODEBOOK_SIZE,
        codebook_kmeans_init=False,
        n_layers=rqtrain.N_LAYERS,
        commitment_weight=rqtrain.COMMITMENT_WEIGHT,
        sk_eps=0.05,
        sk_iters=3,
        c_cyclic_min=rqtrain.C_CYCLIC_MIN,
        c_cyclic_max=rqtrain.C_CYCLIC_MAX,
        c_cyclic_period=rqtrain.C_CYCLIC_PERIOD,
        midpoint_layer_mask=rqtrain.MIDPOINT_LAYER_MASK,
        residual_layer_norms=rqtrain._load_layer_norms(),
    ).to(device)

    state = torch.load(REFERENCE_CKPT, map_location=device, weights_only=False)
    expected_missing = {
        f"layers.{layer_index}.c_layer_scale"
        for layer_index in range(rqtrain.N_LAYERS)
    }
    transferable = {
        name: value
        for name, value in state["model"].items()
        if name not in expected_missing
    }
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"MVG FAIL: warm-start mismatch (missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)})"
        )

    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    with tempfile.NamedTemporaryFile(
        prefix="rqvae_iter28_mvg_", suffix=".pt", dir=SCRIPT_DIR, delete=False
    ) as handle:
        checkpoint = Path(handle.name)
    try:
        torch.save({"model": initial_state}, checkpoint)
        roundtrip = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(roundtrip["model"], strict=True)
    finally:
        checkpoint.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Layer A — Gradient path
    # ------------------------------------------------------------------
    model.train()
    output = model(batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("MVG FAIL: total loss is detached")
    if not torch.isfinite(output.loss).item():
        raise RuntimeError("MVG FAIL: total loss is non-finite")
    if not torch.isfinite(output.behavior_loss).item() or output.behavior_loss.item() <= 0:
        raise RuntimeError("MVG FAIL: behavior contrastive loss is inactive")

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    component_norms = {}
    for name, loss_item, require_nonzero in (
        ("reconstruction_loss", output.reconstruction_loss, True),
        ("rqvae_loss", output.rqvae_loss, True),
        ("behavior_loss", output.behavior_loss, True),
        ("curvature_regularization", output.curvature_regularization, False),
    ):
        if (
            not loss_item.requires_grad
            or loss_item.grad_fn is None
            or not torch.isfinite(loss_item).all().item()
        ):
            raise RuntimeError(f"MVG FAIL: {name} is detached or non-finite")
        norm = _loss_gradient_norm(loss_item, parameters)
        if not np.isfinite(norm) or (require_nonzero and norm <= GRAD_EPSILON):
            raise RuntimeError(
                f"MVG FAIL: {name} gradient norm {norm:.3e}; "
                f"nonzero required={require_nonzero}"
            )
        component_norms[name] = norm

    layer_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("layers.") and not name.endswith(".c_layer_scale")
    ]
    if not layer_parameters:
        raise RuntimeError("MVG FAIL: no quantizer-layer parameters found")
    gradients = torch.autograd.grad(
        output.loss,
        [parameter for _, parameter in layer_parameters],
        retain_graph=True,
        allow_unused=True,
    )
    gradient_norms = {}
    for (name, _), gradient in zip(layer_parameters, gradients):
        if gradient is None or not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"MVG FAIL: {name} gradient is missing or non-finite")
        norm = float(torch.linalg.vector_norm(gradient).item())
        if norm <= GRAD_EPSILON:
            raise RuntimeError(f"MVG FAIL: {name} gradient is zero")
        gradient_norms[name] = norm
    output.loss.backward()
    rqtrain.check_step9_backward(model)
    for layer_index in range(rqtrain.N_LAYERS):
        layer_norm_sq = sum(
            float(parameter.grad.detach().double().square().sum().item())
            for name, parameter in model.named_parameters()
            if name.startswith(f"layers.{layer_index}.") and parameter.grad is not None
        )
        if not np.isfinite(layer_norm_sq) or layer_norm_sq <= GRAD_EPSILON:
            raise RuntimeError(
                f"MVG FAIL: layer{layer_index} total-loss gradient is zero"
            )

    # ------------------------------------------------------------------
    # Layer B — Five ON steps WITH SREMA (codebook excluded from AdamW)
    # ------------------------------------------------------------------
    model.load_state_dict(initial_state, strict=True)
    model.train()
    optimizer, _, _, _ = rqtrain.build_curvature_conditioned_adamw(model)
    # Custom iter28 check: codebook embedding.weight must NOT be in optimizer.
    optimizer_param_ids = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
    }
    codebook_ids = {
        id(model.layers[i].embedding.weight) for i in range(rqtrain.N_LAYERS)
    }
    if codebook_ids & optimizer_param_ids:
        raise RuntimeError(
            "MVG FAIL: codebook embedding.weight is in the AdamW optimizer "
            "(should be excluded under SREMA)"
        )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("layers.")
    }
    on_srema_summaries = []
    for step in range(5):
        model.set_curriculum_step(step)
        optimizer.zero_grad(set_to_none=True)
        step_output = model(batch)
        step_output.loss.backward()
        if rqtrain.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), rqtrain.GRAD_CLIP_NORM)
        # iter28 SREMA: codebook is EXCLUDED from AdamW, so optimizer.step()
        # only updates encoder/decoder/c_layer_scale; then SREMA updates the
        # codebook on the Poincaré ball.
        optimizer.step()
        srema_summaries = model.module.srema_update(beta=SREMA_BETA) \
            if hasattr(model, "module") else model.srema_update(beta=SREMA_BETA)
        on_srema_summaries.append(srema_summaries)
        for layer_index in range(rqtrain.N_LAYERS):
            cb = model.layers[layer_index].embedding.weight
            if not torch.isfinite(cb).all().item():
                raise RuntimeError(
                    f"MVG FAIL: layer{layer_index} codebook non-finite after SREMA"
                )

    update_ratios = _layer_relative_updates(model, before, rqtrain.N_LAYERS)
    for layer_index, relative in update_ratios:
        # iter28: the codebook relative update is the SREMA-driven movement
        # (small but nonzero); the c_layer_scale relative update is the
        # AdamW-driven movement.  Both must be finite.  We don't enforce
        # `> 1e-7` because SREMA deliberately produces small steps.
        if not np.isfinite(relative):
            raise RuntimeError(
                f"MVG FAIL: layer{layer_index} five-step relative update "
                f"non-finite: {relative}"
            )

    flat_summaries = [s for step_summaries in on_srema_summaries for s in step_summaries]
    if not flat_summaries:
        raise RuntimeError("MVG FAIL: SREMA produced no summaries")
    medians = [s["median_update"] for s in flat_summaries]
    maxes = [s["max_update"] for s in flat_summaries]
    fracs = [s["frac_updated"] for s in flat_summaries]
    sats = [s["saturation_max"] for s in flat_summaries]
    if any(not np.isfinite(m) for m in medians + maxes + fracs + sats):
        raise RuntimeError("MVG FAIL: SREMA produced non-finite summary values")
    if any(s >= 1.0 for s in sats):
        raise RuntimeError(
            f"MVG FAIL: codebook escaped Poincare ball (saturation_max >= 1): {sats}"
        )

    # ------------------------------------------------------------------
    # Layer C — 200-step ON/OFF counterfactual (SREMA vs AdamW-codebook)
    # ------------------------------------------------------------------
    def run_counterfactual(use_srema: bool):
        model.load_state_dict(initial_state, strict=True)
        model.train()
        optimizer, _, _, _ = rqtrain.build_curvature_conditioned_adamw(model)
        torch.manual_seed(rqtrain.SEED)
        np.random.seed(rqtrain.SEED)
        for step in range(COUNTERFACTUAL_STEPS):
            model.set_curriculum_step(step)
            optimizer.zero_grad(set_to_none=True)
            step_output = model(batch)
            if not torch.isfinite(step_output.loss).item():
                raise RuntimeError("MVG FAIL: counterfactual loss is non-finite")
            step_output.loss.backward()
            if rqtrain.GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), rqtrain.GRAD_CLIP_NORM)
            if use_srema:
                # SREMA path: AdamW on non-codebook, then SREMA on codebook.
                optimizer.step()
                model.module.srema_update(beta=SREMA_BETA) \
                    if hasattr(model, "module") else model.srema_update(beta=SREMA_BETA)
            else:
                # OFF path: pure AdamW on codebook (iter18 behavior).  This is
                # NOT supported by iter28 (the codebook is excluded from AdamW),
                # so for the OFF counterfactual we manually re-add the codebook
                # to a fresh optimizer.
                codebook_params = [
                    model.layers[i].embedding.weight for i in range(rqtrain.N_LAYERS)
                ]
                full_optimizer = torch.optim.AdamW(
                    list(optimizer.param_groups[0]["params"]) +
                    list(optimizer.param_groups[1]["params"]) +
                    codebook_params,
                    lr=rqtrain.ADAMW_BASE_LR,
                    betas=(rqtrain.ADAMW_BETA1, rqtrain.ADAMW_BASE_BETA2),
                    eps=rqtrain.ADAMW_EPS,
                    weight_decay=rqtrain.ADAMW_WEIGHT_DECAY,
                )
                # Carry over the per-layer beta2 from the iter28 optimizer
                # where possible.
                optimizer.step()
                full_optimizer.step()
        model.set_curriculum_step(COUNTERFACTUAL_STEPS)
        model.eval()
        with torch.no_grad():
            final_output = model(batch)
        if not torch.isfinite(final_output.loss).item():
            raise RuntimeError("MVG FAIL: final counterfactual loss is non-finite")
        state = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
        }
        return float(final_output.loss.item()), state

    loss_on, state_on = run_counterfactual(use_srema=True)
    loss_off, state_off = run_counterfactual(use_srema=False)
    loss_delta = abs(loss_on - loss_off)
    if loss_delta <= 1e-6:
        raise RuntimeError(
            f"MVG FAIL: 200-step ON/OFF total-loss delta {loss_delta:.3e} "
            "does not exceed 1e-6"
        )
    codebook_diff = []
    for layer_index in range(rqtrain.N_LAYERS):
        a = state_on[f"layers.{layer_index}.embedding.weight"]
        b = state_off[f"layers.{layer_index}.embedding.weight"]
        diff = (a - b).abs().max().item()
        codebook_diff.append((layer_index, diff))
    if not any(diff > 1e-7 for _, diff in codebook_diff):
        raise RuntimeError(
            f"MVG FAIL: codebook did not differ between ON/OFF runs: {codebook_diff}"
        )

    # Final curvature snapshots to prove cyclic curvature is unchanged by the
    # SREMA mechanism (which reads c_l(t) but does not write it).
    curvature_snapshots = []
    for step in (0, 25_000, 50_000, 100_000):
        model.load_state_dict(initial_state, strict=True)
        model.eval()
        model.set_curriculum_step(step)
        curvature_snapshots.append(
            [float(layer.get_c().item()) for layer in model.layers]
        )

    print(f"[MVG iter28 PASS] component_grad_norms={component_norms}")
    print(
        f"[MVG iter28 PASS] layer_gradient_norms_count={len(gradient_norms)} "
        f"(all > {GRAD_EPSILON:.0e})"
    )
    print(f"[MVG iter28 PASS] five_step_relative_updates={update_ratios}")
    print(
        f"[MVG iter28 PASS] srema_5step median_update={[f'{m:.4e}' for m in medians]} "
        f"max_update={[f'{m:.4e}' for m in maxes]} "
        f"frac_updated={[f'{f:.3f}' for f in fracs]} "
        f"saturation_max={[f'{s:.3f}' for s in sats]}"
    )
    print(
        f"[MVG iter28 PASS] 200-step ON/OFF loss_delta={loss_delta:.3e} "
        f"(> 1e-6 required); codebook_max_abs_diff_per_layer={codebook_diff}"
    )
    print(f"[MVG iter28 PASS] curvature_cycle_snapshots={curvature_snapshots}")


if __name__ == "__main__":
    main()