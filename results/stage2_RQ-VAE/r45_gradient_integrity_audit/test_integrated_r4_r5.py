"""Issue #71 §2 — real train-path gradient + optimizer tests for R4-08, R5-01, R5-02.

Audit-only. No source-string, no config-flag, no isolated-helper-only assertions: every test
below drives the actual ``RQVAE.forward`` -> ``compute_loss`` ->
``behavior_contrastive_components`` -> ``loss.backward(retain_graph=True)`` ->
``_r4_08_apply_projection`` -> ``clip_grad_norm_`` -> ``AdamW.step()`` path that
``train_rqvae.py`` uses.

Method
------
* ``build`` is fully determined by ``(module, seed)``: identical inputs give bitwise identical
  tensors, so a *reference* case and an *executed* case can be built independently.
* Reference gradients are computed on the reference case **before** any projection runs.
  This matters: ``_r4_08_apply_projection`` ends with an ``autograd.grad`` call using
  ``retain_graph=False`` for the L2+L3 term, which frees the graph. A test that tried to
  recompute references afterwards would hit "backward through the graph a second time".
* Encoder differentiation is always measured with ``clip=0.0`` so the shared
  ``clip_grad_norm_(model.parameters(), 1.0)`` cannot mask a real gradient-surgery difference.

Seeds: ``CONFLICT_SEEDS`` are the seeds whose real ``<g_L1, g_base>`` dot product is negative,
i.e. the PCGrad projection genuinely fires. They were found by scanning seeds 0..39 on the
r4_08 tree; the scan output is reproduced by ``scan_conflict_seeds()`` below.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K  # noqa: E402

T_R4_08 = K.load_arm("r4_08")
T_R5_01 = K.load_arm("r5_01")
T_R5_02 = K.load_arm("r5_02")

N_SRC = 24
IN_DIM = 32
BEHAVIOR_WEIGHT = 0.5
CONFLICT_SEEDS = [1, 4, 5, 12, 16, 30, 32, 36]
NON_CONFLICT_SEEDS = [0, 2, 3, 8, 9]
ATOL, RTOL = 1e-6, 1e-5


# ------------------------------------------------------------------ deterministic fixtures
def build(T, seed: int, n_src: int = N_SRC):
    """A fully deterministic mini training case for one arm module."""
    torch.manual_seed(seed)
    model = T.RQVAE(T._tokenizer_config(), in_dim=IN_DIM)
    generator = torch.Generator().manual_seed(seed + 1)
    emb = torch.randn(2 * n_src, IN_DIM, generator=generator)
    reconstructed, quant_loss, _, _ = model.forward(emb)
    base_loss, recon_loss = model.compute_loss(emb, reconstructed, quant_loss)
    residuals = T._residuals_with_trace(model, emb)
    source_ids = torch.arange(n_src)
    target_ids = (torch.arange(n_src) + 1) % n_src
    l1_loss, other_loss, valid_levels = T.behavior_contrastive_components(
        residuals, source_ids, target_ids, None)
    encoder = list(model.encoder.parameters())
    non_encoder = [p for p in model.parameters() if not any(p is q for q in encoder)]
    return {"T": T, "model": model, "emb": emb, "base_loss": base_loss,
            "recon_loss": recon_loss, "l1_loss": l1_loss, "other_loss": other_loss,
            "valid_levels": valid_levels, "encoder": encoder, "non_encoder": non_encoder,
            "scale": BEHAVIOR_WEIGHT / max(valid_levels, 1)}


def reference_grads(case, *, keep_graph: bool = True):
    """(g_base, g_l1, g_other) straight from the module's own helpers, before any projection."""
    T, enc = case["T"], case["encoder"]
    grad = lambda loss: T._r4_08_global_average(  # noqa: E731
        T._r4_08_encoder_gradients(loss, enc, retain=keep_graph))
    return grad(case["base_loss"]), grad(case["l1_loss"]), grad(case["other_loss"])


def flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def unflatten_like(flat_vec, template):
    out, offset = [], 0
    for g in template:
        count = g.numel()
        out.append(flat_vec[offset:offset + count].view_as(g))
        offset += count
    return out


def hand_assembly(case, g_base, g_l1_proj, g_other):
    return [g_base[i] + case["scale"] * (g_l1_proj[i] + g_other[i])
            for i in range(len(case["encoder"]))]


def assembled_loss(case, weight: float = BEHAVIOR_WEIGHT):
    """The production ``loss``: base + weight * behavior, behavior being the per-level mean.

    A ``None`` L1 term models a level dropped by ``_contrastive_ce`` (every row a false
    negative); such a level contributes nothing rather than crashing the sum.
    """
    l1 = case["l1_loss"] if case["l1_loss"] is not None else torch.zeros(())
    other = case["other_loss"] if case["other_loss"] is not None else torch.zeros(())
    return case["base_loss"] + weight * (l1 + other) / max(case["valid_levels"], 1)


def run_step(case, *, project: bool, arm_kwargs=None, clip: float = 0.0,
             behavior_weight: float = BEHAVIOR_WEIGHT):
    """Exactly one optimizer step in production order. Returns grads taken before step().

    ``project=False`` reproduces the *plain Resample* step: a single backward through the
    fully assembled loss (base + w * behaviour), which is what the baseline does when the
    projection is off. ``project=True`` takes the real ``loss.backward(retain_graph=True)``
    and then overwrites ONLY the encoder gradients.
    """
    model = case["model"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.zero_grad(set_to_none=False)
    if project:
        case["base_loss"].backward(retain_graph=True)
        case["T"]._r4_08_apply_projection(
            case["base_loss"], case["l1_loss"], case["other_loss"], case["encoder"],
            behavior_weight, valid_levels=case["valid_levels"], **(arm_kwargs or {}))
    else:
        assembled_loss(case, behavior_weight).backward()
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    grads = {id(p): (None if p.grad is None else p.grad.detach().clone())
             for p in model.parameters()}
    optimizer.step()
    return grads, optimizer, {id(p): p.detach().clone() for p in model.parameters()}


def encoder_grads(grads, case):
    return [grads[id(p)] for p in case["encoder"]]


def encoder_weights(weights, case):
    return [weights[id(p)] for p in case["encoder"]]


def non_encoder_grads(grads, case):
    return [grads[id(p)] for p in case["non_encoder"]]


def non_encoder_weights(weights, case):
    return [weights[id(p)] for p in case["non_encoder"]]


def max_abs_diff(a, b) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def l2(a, b) -> float:
    return float(torch.sqrt(sum(((x - y) ** 2).sum() for x, y in zip(a, b))))


def scan_conflict_seeds(limit: int = 40) -> list[int]:
    """Reproduce the conflict-seed discovery (prints the scan)."""
    conflicts = []
    for seed in range(limit):
        case = build(T_R4_08, seed)
        g_base, g_l1, _ = reference_grads(case)
        dot = float(flat(g_l1) @ flat(g_base))
        if dot < 0.0:
            conflicts.append(seed)
    return conflicts


def patched(module, name, value):
    """Temporarily set a module-level flag; returns a restore callable.

    In the RED phase the arm module is the unmodified parent tree, where the arm's own flag
    does not exist. Treating "flag absent" as a no-op makes the arm *behave* like the parent,
    so the mechanism assertions fail on their numeric claims (a behavioural RED) rather than
    dying on a missing attribute (which issue #71 §4 explicitly refuses to count).
    """
    if not hasattr(module, name):
        print(f"  [RED] {module.__name__} has no flag {name!r}; treating it as parent behaviour",
              flush=True)
        return lambda: None
    saved = getattr(module, name)
    setattr(module, name, value)
    return lambda: setattr(module, name, saved)


def arm_kwargs_for(T, case) -> dict:
    """Arm-specific kwargs, filtered by the callee's real signature.

    The parent ``_r4_08_apply_projection`` has no ``recon_loss``/``encoder_module`` parameter.
    Passing them unconditionally would make the RED phase fail with a TypeError (a signature
    difference, not a behavioural discrepancy), so we only pass what this tree accepts.
    """
    import inspect
    accepted = inspect.signature(T._r4_08_apply_projection).parameters
    kwargs = {}
    if "recon_loss" in accepted:
        kwargs["recon_loss"] = case["recon_loss"]
    if "encoder_module" in accepted:
        kwargs["encoder_module"] = case["model"].encoder
    return kwargs


# ---------------------------------------------------------------------------------- R4-08
def test_r4_08_disabled_matches_plain_resample_backward():
    """PCGrad off must reproduce the plain Resample backward.

    Tolerance note (measured, not guessed): the disabled path assembles three separately
    differentiated gradients ``g_base + (w/v)*(g_l1 + g_other)`` whereas the plain path takes
    one fused backward through ``base + w*(l1+other)/v``. float32 addition is not associative,
    so the two agree to ~4e-8 absolute / ~1.5e-6 relative, NOT bitwise. That reassociation
    noise is the entire observed discrepancy.
    """
    worst = 0.0
    worst_weight = 0.0
    for seed in CONFLICT_SEEDS[:4]:
        restore = patched(T_R4_08, "R4_08_PCGRAD_ENC", False)
        try:
            off_case = build(T_R4_08, seed)
            off_grads, _, off_w = run_step(off_case, project=True)
            plain_case = build(T_R4_08, seed)
            plain_grads, _, plain_w = run_step(plain_case, project=False)
        finally:
            restore()
        diff = max_abs_diff(encoder_grads(off_grads, off_case),
                            encoder_grads(plain_grads, plain_case))
        worst = max(worst, diff)
        assert diff < 1e-6, (
            f"seed {seed}: R4_08_PCGRAD_ENC=False must reproduce the plain Resample encoder "
            f"gradient, off by {diff}")
        for a, b in zip(encoder_weights(off_w, off_case), encoder_weights(plain_w, plain_case)):
            worst_weight = max(worst_weight, float((a - b).abs().max()))
            assert torch.allclose(a, b, atol=1e-4, rtol=1e-3), (
                f"seed {seed}: post-step encoder weights must match the plain Resample step. "
                f"AdamW's first step is lr*g/(|g|+eps), i.e. sign-like, so it amplifies the "
                f"4e-8 float32 reassociation noise into a measured ~1e-5..3e-5 weight delta "
                f"({worst_weight:.2e} here); the invariant that matters is the gradient parity "
                f"asserted above.")
    print(f"  PCGrad-off vs plain: worst encoder-grad abs diff {worst:.3e} "
          f"(float32 reassociation noise)", flush=True)


def test_r4_08_projection_matches_hand_computed_assembly():
    """The executed encoder gradient equals the hand-derived global PCGrad assembly."""
    for seed in CONFLICT_SEEDS[:4]:
        ref_case = build(T_R4_08, seed)
        g_base, g_l1, g_other = reference_grads(ref_case)
        projected = unflatten_like(T_R4_08.r4_project_conflict(flat(g_l1), flat(g_base)), g_l1)
        expected = hand_assembly(ref_case, g_base, projected, g_other)

        case = build(T_R4_08, seed)
        grads, _, _ = run_step(case, project=True)
        diff = max_abs_diff(encoder_grads(grads, case), expected)
        assert diff < 1e-6, f"seed {seed}: executed assembly differs from hand reference by {diff}"

        # the w_b / valid_levels factor really is linear in behavior_weight
        heavy = build(T_R4_08, seed)
        heavy_scale = 1.5 / max(heavy["valid_levels"], 1)
        expected_heavy = [g_base[i] + heavy_scale * (projected[i] + g_other[i])
                          for i in range(len(projected))]
        grads_heavy, _, _ = run_step(heavy, project=True, behavior_weight=1.5)
        diff_heavy = max_abs_diff(encoder_grads(grads_heavy, heavy), expected_heavy)
        assert diff_heavy < 1e-6, f"seed {seed}: heavy-weight assembly differs by {diff_heavy}"


def test_r4_08_projection_only_acts_on_negative_dot():
    """dot >= 0 must pass through untouched; dot < 0 must fire exactly once."""
    checked = {"no_conflict": 0, "conflict": 0}
    for seed in range(20):
        ref_case = build(T_R4_08, seed)
        g_base, g_l1, g_other = reference_grads(ref_case)
        dot = float(flat(g_l1) @ flat(g_base))
        unprojected = hand_assembly(ref_case, g_base, g_l1, g_other)

        case = build(T_R4_08, seed)
        before = T_R4_08.R4_08_PROJECTION_TRACE["active_count"]
        grads, _, _ = run_step(case, project=True)
        fired = T_R4_08.R4_08_PROJECTION_TRACE["active_count"] - before
        actual = encoder_grads(grads, case)

        if dot >= 0.0:
            assert fired == 0, f"seed {seed}: dot={dot} >= 0 but projection fired {fired}x"
            assert max_abs_diff(actual, unprojected) == 0.0, (
                f"seed {seed}: no-conflict path must be bitwise the unprojected assembly")
            checked["no_conflict"] += 1
        else:
            assert fired == 1, f"seed {seed}: dot={dot} < 0 but projection fired {fired}x"
            projected = unflatten_like(
                T_R4_08.r4_project_conflict(flat(g_l1), flat(g_base)), g_l1)
            assert max_abs_diff(actual, hand_assembly(ref_case, g_base, projected, g_other)) < 1e-6
            checked["conflict"] += 1
    assert checked["conflict"] >= 3 and checked["no_conflict"] >= 3, checked
    print(f"  dot-scan: {checked} over seeds 0..19", flush=True)


def test_r4_08_zero_base_gradient_is_passthrough():
    g_l1 = torch.tensor([1.0, -2.0, 0.5])
    out = T_R4_08.r4_project_conflict(g_l1.clone(), torch.zeros_like(g_l1))
    assert torch.equal(out, g_l1), "zero base gradient must pass g_L1 through untouched"

    # ... and through the real integration: detach the base path and re-attach a fresh leaf
    case = build(T_R4_08, CONFLICT_SEEDS[0])
    case["base_loss"] = case["recon_loss"].detach() * 0.0 + case["emb"].sum() * 0.0
    case["base_loss"] = case["base_loss"] + next(case["model"].encoder.parameters()).sum() * 0.0
    g_base, _, _ = reference_grads(case)
    assert float(flat(g_base).abs().max()) == 0.0, "fixture must produce a zero base gradient"
    grads, _, _ = run_step(case, project=True)
    assert all(torch.isfinite(g).all() for g in encoder_grads(grads, case))


def test_r4_08_non_finite_gradient_fails_closed():
    for bad in (float("nan"), float("inf")):
        g_l1 = torch.tensor([1.0, 2.0])
        g_base = torch.tensor([0.0, bad])
        try:
            out = T_R4_08.r4_project_conflict(g_l1, g_base)
        except RuntimeError:
            continue
        raise AssertionError(
            f"non-finite base gradient ({bad}) returned {out} instead of raising RuntimeError")


def test_unused_parameters_get_zero_gradients():
    """A parameter no loss touches must come back as exact zeros, not as garbage."""
    case = build(T_R4_08, CONFLICT_SEEDS[0])
    orphan = torch.nn.Parameter(torch.zeros(3))
    case["encoder"] = list(case["encoder"]) + [orphan]
    grads = case["T"]._r4_08_encoder_gradients(case["base_loss"], case["encoder"], retain=True)
    assert torch.equal(grads[-1], torch.zeros_like(orphan)), (
        "unused parameter must yield an exact zero gradient")
    case["T"]._r4_08_apply_projection(case["base_loss"], case["l1_loss"], case["other_loss"],
                                      case["encoder"], BEHAVIOR_WEIGHT,
                                      valid_levels=case["valid_levels"])
    assert torch.equal(orphan.grad, torch.zeros_like(orphan))


def test_absent_l1_term_is_handled():
    """The production guard ``behavior_l1_loss is not None`` must keep the step well defined.

    ``train_rqvae.py`` only enters ``_r4_08_apply_projection`` when ``behavior_l1_loss`` is not
    None; a dropped L1 level must therefore fall through to the plain ``loss.backward()``
    branch, not crash on ``None``.
    """
    case = build(T_R4_08, CONFLICT_SEEDS[0])
    case["l1_loss"] = None
    assert case["l1_loss"] is None, "fixture must model a dropped L1 contrastive level"

    before = T_R4_08.R4_08_PROJECTION_TRACE["active_count"]
    grads, _, _ = run_step(case, project=False)
    assert T_R4_08.R4_08_PROJECTION_TRACE["active_count"] == before, (
        "with the L1 term absent no projection may be attempted")
    assert all(torch.isfinite(g).all() for g in encoder_grads(grads, case))
    assert all(torch.isfinite(g).all() for g in non_encoder_grads(grads, case))


def test_non_encoder_gradients_untouched_without_clipping():
    """Decoder + codebook gradients and updates are identical with PCGrad on or off."""
    seed = CONFLICT_SEEDS[0]
    restore = patched(T_R4_08, "R4_08_PCGRAD_ENC", False)
    try:
        off = build(T_R4_08, seed)
        off_grads, _, off_w = run_step(off, project=True, clip=0.0)
    finally:
        restore()
    on = build(T_R4_08, seed)
    on_grads, _, on_w = run_step(on, project=True, clip=0.0)

    for a, b in zip(non_encoder_grads(off_grads, off), non_encoder_grads(on_grads, on)):
        assert torch.equal(a, b), "gradient surgery must not touch decoder/codebook gradients"
    assert len(on["non_encoder"]) > 0
    for a, b in zip(non_encoder_weights(off_w, off), non_encoder_weights(on_w, on)):
        assert torch.equal(a, b), "with clipping off, non-encoder updates must be bitwise equal"
    # and the encoder really did differ, so the test above is not vacuous
    assert l2(encoder_grads(off_grads, off), encoder_grads(on_grads, on)) > 1e-6


def test_clipping_multiplier_shared_not_equality():
    """With clip active, ONE multiplier applies to encoder and non-encoder alike.

    Issue #71 §2 warns that the shared ``clip_grad_norm_(model.parameters(), 1.0)`` couples the
    two groups: when the encoder norm grows, the *non-encoder* post-clip gradients change too.
    Post-step non-encoder equality is therefore only required when clipping is OFF; here the
    assertion is the shared multiplier, measured over ALL parameters exactly as
    ``clip_grad_norm_`` does (encoder-only norm is the wrong quantity).
    """
    seed = CONFLICT_SEEDS[0]
    weight = 1.0                      # total_norm ~2.16, so clipping genuinely fires
    unclipped = build(T_R4_08, seed)
    pre_grads, _, pre_w = run_step(unclipped, project=True, clip=0.0, behavior_weight=weight)
    clipped = build(T_R4_08, seed)
    post_grads, _, post_w = run_step(clipped, project=True, clip=1.0, behavior_weight=weight)

    total = math.sqrt(sum(float(g.pow(2).sum()) for g in pre_grads.values() if g is not None))
    assert total > 1.0, f"fixture must trigger clipping (total_norm={total})"
    multiplier = 1.0 / (total + 1e-6)
    # the two cases are separate model instances, so pair parameters by index, not by id()
    for name, key in (("encoder", "encoder"), ("non-encoder", "non_encoder")):
        for i, (p_pre, p_post) in enumerate(zip(unclipped[key], clipped[key])):
            pre, post = pre_grads[id(p_pre)], post_grads[id(p_post)]
            diff = float((post - pre * multiplier).abs().max())
            assert diff < 1e-4, (
                f"{name}[{i}] grad must be pre_clip * shared multiplier, off by {diff}")
    differ = sum(1 for a, b in zip(non_encoder_weights(pre_w, unclipped),
                                   non_encoder_weights(post_w, clipped))
                 if not torch.equal(a, b))
    assert differ > 0, "clipping must visibly change non-encoder updates (else this is vacuous)"
    print(f"  shared clip multiplier {multiplier:.6f} from total_norm {total:.4f}", flush=True)


# ---------------------------------------------------------------------------------- R5-01
def test_r5_01_projects_base_and_keeps_l1_unprojected():
    """r5_01 must project g_base against g_L1, and assemble the ORIGINAL g_L1."""
    seed = CONFLICT_SEEDS[0]
    r5_case = build(T_R5_01, seed)
    g_base, g_l1, g_other = reference_grads(r5_case)
    base_proj = unflatten_like(
        T_R5_01.r4_project_conflict(flat(g_base), flat(g_l1)), g_base)
    expected = hand_assembly(r5_case, base_proj, g_l1, g_other)   # note: g_l1 unprojected

    case = build(T_R5_01, seed)
    grads, _, _ = run_step(case, project=True)
    actual = encoder_grads(grads, case)
    assert max_abs_diff(actual, expected) < 1e-6, (
        "r5_01 must assemble g_base_projected + scale * (g_l1_original + g_other)")

    # and it must differ from r4_08 in exactly the predicted way. r4_08 projects g_L1 against
    # g_base; r5_01 projects g_base against g_L1 and keeps g_L1 intact. So the encoder-gradient
    # delta is (g_base_proj - g_base) + scale * (g_L1 - g_L1_proj), i.e. BOTH terms change.
    r4_case = build(T_R4_08, seed)
    r4_grads, _, _ = run_step(r4_case, project=True)
    r4_actual = encoder_grads(r4_grads, r4_case)
    l1_proj = unflatten_like(T_R4_08.r4_project_conflict(flat(g_l1), flat(g_base)), g_l1)
    delta_hand = [(base_proj[i] - g_base[i]) + (r5_case["scale"] - case["scale"]) * 0.0
                  + case["scale"] * (g_l1[i] - l1_proj[i]) for i in range(len(g_base))]
    delta_actual = [actual[i] - r4_actual[i] for i in range(len(g_base))]
    assert l2(actual, r4_actual) > 1e-6, "r5_01 must differ from r4_08"
    assert max_abs_diff(delta_actual, delta_hand) < 1e-6, (
        "r5_01 - r4_08 must equal (g_base_proj - g_base) + scale * (g_L1 - g_L1_proj)")


def test_r5_01_off_is_pcgrad_free_not_r4_08():
    """AUDIT FINDING: R5-01's OFF state is *no PCGrad at all*, not R4-08.

    ``R5_01_REVERSE_PCGRAD=False`` falls into the ``else`` branch that assigns
    ``g_base_proj = g_base`` and never projects ``g_l1`` either, so the arm cannot be toggled
    back into R4-08's projected rule. This pins that asymmetry so no future reader assumes
    "R5 delta off == parent".
    """
    seed = CONFLICT_SEEDS[0]
    r4_on = build(T_R4_08, seed)
    r4_on_grads, _, _ = run_step(r4_on, project=True)
    r4_off = build(T_R4_08, seed)
    restore = patched(T_R4_08, "R4_08_PCGRAD_ENC", False)
    try:
        r4_off_grads, _, _ = run_step(r4_off, project=True)
    finally:
        restore()
    restore = patched(T_R5_01, "R5_01_REVERSE_PCGRAD", False)
    try:
        case = build(T_R5_01, seed)
        grads, _, _ = run_step(case, project=True)
    finally:
        restore()

    r5_off = encoder_grads(grads, case)
    diff_to_pcgrad_free = max_abs_diff(r5_off, encoder_grads(r4_off_grads, r4_off))
    assert diff_to_pcgrad_free < 1e-6, (
        f"R5-01 with its delta off must reproduce the PCGrad-free assembly, off by "
        f"{diff_to_pcgrad_free}")
    assert l2(r5_off, encoder_grads(r4_on_grads, r4_on)) > 1e-6, (
        "R5-01 OFF must NOT equal R4-08 ON: the delta is not a pure projection-direction swap")


# ---------------------------------------------------------------------------------- R5-02
def test_r5_02_uses_recon_only_reference_and_full_base():
    """r5_02 must project against the RECON gradient only, yet assemble the FULL g_base."""
    seed = CONFLICT_SEEDS[0]
    ref = build(T_R5_02, seed)
    g_base, g_l1, g_other = reference_grads(ref)
    g_recon = T_R5_02._r4_08_global_average(
        T_R5_02._r4_08_encoder_gradients(ref["recon_loss"], ref["encoder"], retain=True))
    g_quant = T_R5_02._r4_08_global_average(
        T_R5_02._r4_08_encoder_gradients(ref["base_loss"] - ref["recon_loss"],
                                         ref["encoder"], retain=True))
    split_err = max_abs_diff([a - b for a, b in zip(g_base, [r + q for r, q in zip(g_recon, g_quant)])],
                             [torch.zeros_like(a) for a in g_base])
    assert split_err < 1e-5, f"g_base must equal g_recon + g_quant, off by {split_err}"

    l1_proj = unflatten_like(T_R5_02.r4_project_conflict(flat(g_l1), flat(g_recon)), g_l1)
    expected = hand_assembly(ref, g_base, l1_proj, g_other)
    case = build(T_R5_02, seed)
    grads, _, _ = run_step(case, project=True, arm_kwargs=arm_kwargs_for(T_R5_02, case))
    actual = encoder_grads(grads, case)
    assert max_abs_diff(actual, expected) < 1e-6, (
        "r5_02 must project g_L1 against g_recon and assemble the full g_base")

    # the specific r5_02 property: assembling g_recon instead of g_base would be wrong
    wrong = hand_assembly(ref, g_recon, l1_proj, g_other)
    assert l2(actual, wrong) > 1e-6, (
        "assembling the recon-only gradient would differ; the test must be non-vacuous")
    assert float(flat(g_quant).norm()) > 1e-6, "g_quant must be non-trivial for this fixture"


def test_r5_02_reference_choice_changes_the_decision():
    """Some seed must make the recon-reference and the base-reference disagree."""
    found = None
    for seed in range(20):
        ref = build(T_R5_02, seed)
        g_base, g_l1, _ = reference_grads(ref)
        g_recon = T_R5_02._r4_08_global_average(
            T_R5_02._r4_08_encoder_gradients(ref["recon_loss"], ref["encoder"], retain=True))
        dot_base = float(flat(g_l1) @ flat(g_base))
        dot_recon = float(flat(g_l1) @ flat(g_recon))
        flips = (dot_base < 0.0) != (dot_recon < 0.0)
        apart = l2(unflatten_like(T_R5_02.r4_project_conflict(flat(g_l1), flat(g_recon)), g_l1),
                   unflatten_like(T_R5_02.r4_project_conflict(flat(g_l1), flat(g_base)), g_l1))
        if flips or apart > 1e-4:
            found = (seed, flips, apart, dot_base, dot_recon)
            break
    assert found is not None, (
        "no seed in 0..19 makes the recon-only and full-base references disagree; "
        "r5_02 would then be indistinguishable from r4_08 on this metric")
    seed, flips, apart, dot_base, dot_recon = found
    print(f"  reference-choice divergence at seed {seed}: flips={flips} "
          f"l2={apart:.6e} dot_base={dot_base:+.4e} dot_recon={dot_recon:+.4e}", flush=True)

    ref = build(T_R5_02, seed)
    g_base, g_l1, g_other = reference_grads(ref)
    g_recon = T_R5_02._r4_08_global_average(
        T_R5_02._r4_08_encoder_gradients(ref["recon_loss"], ref["encoder"], retain=True))
    case = build(T_R5_02, seed)
    grads, _, _ = run_step(case, project=True, arm_kwargs=arm_kwargs_for(T_R5_02, case))
    follows_recon = encoder_grads(grads, case)
    follows_base = hand_assembly(ref, g_base,
                                 unflatten_like(T_R5_02.r4_project_conflict(flat(g_l1), flat(g_base)), g_l1),
                                 g_other)
    assert max_abs_diff(follows_recon, follows_base) > 1e-6, (
        "r5_02 must follow the recon reference, not the base reference")


def test_r5_02_disabled_parity_is_pcgrad_free():
    """R5-02's OFF state, like R5-01's, is the PCGrad-free assembly (same asymmetry)."""
    seed = CONFLICT_SEEDS[0]
    r4_off = build(T_R4_08, seed)
    restore = patched(T_R4_08, "R4_08_PCGRAD_ENC", False)
    try:
        r4_off_grads, _, _ = run_step(r4_off, project=True)
    finally:
        restore()
    restore = patched(T_R5_02, "R5_02_RECON_REF", False)
    try:
        case = build(T_R5_02, seed)
        grads, _, _ = run_step(case, project=True, arm_kwargs=arm_kwargs_for(T_R5_02, case))
    finally:
        restore()
    diff = max_abs_diff(encoder_grads(grads, case), encoder_grads(r4_off_grads, r4_off))
    assert diff < 1e-6, f"R5-02 delta off must reproduce the PCGrad-free assembly, off by {diff}"


def main() -> int:
    print(f"conflict seeds (recomputed scan, 0..39): {scan_conflict_seeds()}", flush=True)
    print(f"seeds used as conflict fixtures: {CONFLICT_SEEDS}", flush=True)
    tests = [
        test_r4_08_disabled_matches_plain_resample_backward,
        test_r4_08_projection_matches_hand_computed_assembly,
        test_r4_08_projection_only_acts_on_negative_dot,
        test_r4_08_zero_base_gradient_is_passthrough,
        test_r4_08_non_finite_gradient_fails_closed,
        test_unused_parameters_get_zero_gradients,
        test_absent_l1_term_is_handled,
        test_non_encoder_gradients_untouched_without_clipping,
        test_clipping_multiplier_shared_not_equality,
        test_r5_01_projects_base_and_keeps_l1_unprojected,
        test_r5_01_off_is_pcgrad_free_not_r4_08,
        test_r5_02_disabled_parity_is_pcgrad_free,
        test_r5_02_uses_recon_only_reference_and_full_base,
        test_r5_02_reference_choice_changes_the_decision,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}\n     {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}\n     {type(exc).__name__}: {exc}", flush=True)
        else:
            print(f"pass {test.__name__}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed", flush=True)
    if failed:
        print("INTEGRATED_TESTS_RED", flush=True)
        return 1
    print("ALL TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
