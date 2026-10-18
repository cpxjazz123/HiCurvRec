"""Issue #71 §2 — mechanism-specific numerical counterexamples for R5-03 … R5-10.

Audit-only. Each arm gets a counterexample that provably differentiates its **actual integrated
encoder gradient** from the R4-08 parent on the same inputs, the difference is *explained* by
re-deriving that arm's rule by hand, and the arm's OFF state is pinned.

Everything runs through the real ``RQVAE.forward`` -> ``compute_loss`` ->
``behavior_contrastive_components`` -> ``loss.backward(retain_graph=True)`` ->
``_r4_08_apply_projection`` -> ``AdamW.step()`` path. Clipping is disabled (``clip=0.0``) for
every differentiation measurement so the shared ``clip_grad_norm_`` cannot mask a real
gradient-surgery difference.

Determinism
-----------
``build`` is fully determined by ``(module, seed)``. Two seeds are needed per arm: a *reference*
case for hand-derived gradients and an *executed* case, because ``_r4_08_apply_projection``
frees the graph on its ``other_loss`` gradient.

Conflict seeds (real ``<g_L1, g_base>`` < 0 on the R4-08 tree): 1, 4, 5, 12, 16, 30, 32, 36.

Module-global state
-------------------
``r5_09`` keeps ``R5_09_EMA_STATE["buffers"]`` at module scope, so it leaks across cases unless
explicitly reset. ``reset_ema()`` does that and restores it afterwards. This leak is itself a
testability finding and is recorded in the audit report.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K  # noqa: E402

ARMS = [f"r5_{i:02d}" for i in range(3, 11)]
T_R4_08 = K.load_arm("r4_08")
MODULES = {arm: K.load_arm(arm) for arm in ARMS}
T_R5_09 = MODULES["r5_09"]

N_SRC = 24
IN_DIM = 32
BEHAVIOR_WEIGHT = 0.5
CONFLICT_SEEDS = [1, 4, 5, 12, 16, 30, 32, 36]
SEED_PER_ARM = {arm: CONFLICT_SEEDS[i % len(CONFLICT_SEEDS)] for i, arm in enumerate(ARMS)}


# ------------------------------------------------------------------ deterministic fixtures
def build(T, seed: int, n_src: int = N_SRC):
    torch.manual_seed(seed)
    model = T.RQVAE(T._tokenizer_config(), in_dim=IN_DIM)
    emb = torch.randn(2 * n_src, IN_DIM, generator=torch.Generator().manual_seed(seed + 1))
    reconstructed, quant_loss, _, _ = model.forward(emb)
    base_loss, recon_loss = model.compute_loss(emb, reconstructed, quant_loss)
    residuals = T._residuals_with_trace(model, emb)
    l1_loss, other_loss, valid_levels = T.behavior_contrastive_components(
        residuals, torch.arange(n_src), (torch.arange(n_src) + 1) % n_src, None)
    encoder = list(model.encoder.parameters())
    return {"T": T, "model": model, "base_loss": base_loss, "recon_loss": recon_loss,
            "l1_loss": l1_loss, "other_loss": other_loss, "valid_levels": valid_levels,
            "encoder": encoder,
            "non_encoder": [p for p in model.parameters() if not any(p is q for q in encoder)],
            "scale": BEHAVIOR_WEIGHT / max(valid_levels, 1)}


def reference_grads(case, loss: str = "base_loss"):
    T, enc = case["T"], case["encoder"]
    grad = lambda key: T._r4_08_global_average(  # noqa: E731
        T._r4_08_encoder_gradients(case[key], enc, retain=True))
    return grad(loss), grad("l1_loss"), grad("other_loss")


def flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def unflatten_like(flat_vec, template):
    out, offset = [], 0
    for g in template:
        count = g.numel()
        out.append(flat_vec[offset:offset + count].view_as(g))
        offset += count
    return out


def assemble(case, g_base, g_l1_proj, g_other):
    return [g_base[i] + case["scale"] * (g_l1_proj[i] + g_other[i])
            for i in range(len(case["encoder"]))]


def arm_kwargs_for(T, case) -> dict:
    accepted = inspect.signature(T._r4_08_apply_projection).parameters
    kwargs = {}
    if "recon_loss" in accepted:
        kwargs["recon_loss"] = case["recon_loss"]
    if "encoder_module" in accepted:
        kwargs["encoder_module"] = case["model"].encoder
    return kwargs


def assembled_loss(case, weight: float = BEHAVIOR_WEIGHT):
    """The production loss: base + weight * behaviour, behaviour being the per-level mean."""
    l1 = case["l1_loss"] if case["l1_loss"] is not None else torch.zeros(())
    other = case["other_loss"] if case["other_loss"] is not None else torch.zeros(())
    return case["base_loss"] + weight * (l1 + other) / max(case["valid_levels"], 1)


def run_step(case, *, project=True, clip=0.0, epoch=None, skip_projection=False):
    """One optimizer step in production order; returns encoder gradients before the step.

    When the projection branch is not taken -- ``skip_projection``, ``project=False``, or an
    arm whose epoch gate is closed (R5-08 before 2250) -- production takes the plain
    ``else: loss.backward()`` branch, and that ``loss`` is the FULLY ASSEMBLED one, not the
    base term. Backwarding ``base_loss`` alone here would silently drop the behaviour term.
    """
    model = case["model"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.zero_grad(set_to_none=False)
    T = case["T"]
    gate = not (skip_projection or not project)
    if epoch is not None and hasattr(T, "R5_08_LATE_START_EPOCH"):
        gate = gate and epoch >= T.R5_08_LATE_START_EPOCH
    if gate:
        case["base_loss"].backward(retain_graph=True)
        T._r4_08_apply_projection(case["base_loss"], case["l1_loss"], case["other_loss"],
                                  case["encoder"], BEHAVIOR_WEIGHT,
                                  valid_levels=case["valid_levels"],
                                  **arm_kwargs_for(T, case))
    else:
        assembled_loss(case).backward()
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    grads = [None if p.grad is None else p.grad.detach().clone() for p in case["encoder"]]
    optimizer.step()
    return grads, optimizer


def reference_step(seed: int):
    """The R4-08 parent reference for the same inputs/settings."""
    return run_step(build(T_R4_08, seed), project=True)[0]


def max_abs_diff(a, b) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def l2(a, b) -> float:
    return float(torch.sqrt(sum(((x - y) ** 2).sum() for x, y in zip(a, b))))


def patched(module, name, value):
    if not hasattr(module, name):
        print(f"  [RED] {name} absent on {module.EXPERIMENT_ARM}; using parent behaviour",
              flush=True)
        return lambda: None
    saved = getattr(module, name)
    setattr(module, name, value)
    return lambda: setattr(module, name, saved)


def arm_attr(T, name, parent_default):
    """Read an arm constant, substituting the PARENT's semantics when it is absent.

    Without this the RED phase dies on ``AttributeError`` for a missing constant, which issue
    #71 §4 explicitly refuses to count as a behavioural RED: the assertion must land on a
    numeric claim instead. ``parent_default`` is what the unmodified parent tree effectively
    does (e.g. no alpha override -> alpha 1; no cosine gate -> project on any negative dot).
    """
    return getattr(T, name, parent_default)


def ema_state(T):
    """The R5-09 EMA dict, or None when the tree has no EMA at all."""
    return getattr(T, "R5_09_EMA_STATE", None)


def reset_ema(T):
    state = ema_state(T)
    if state is None:
        return lambda: None
    saved = state["buffers"]
    state["buffers"] = None
    return lambda: state.__setitem__("buffers", saved)


def assert_differentiated(arm: str, actual, ref, minimum: float = 1e-5) -> float:
    distance = l2(actual, ref)
    assert distance > minimum, (
        f"{arm}: encoder gradient must differ from the R4-08 reference by > {minimum}; "
        f"measured {distance:.6e}. The counterexample would be vacuous.")
    print(f"  {arm}: ||grad - grad_r4_08||_2 = {distance:.6e}", flush=True)
    return distance


def assert_pcgrad_free_parity(arm: str, T, flag: str, seed: int, **kwargs):
    """With the arm's own delta disabled, BOTH sides must reduce to the PCGrad-free assembly.

    ``g_base + (w/valid_levels) * (g_L1 + g_other)``. The helper disables the parent's
    ``R4_08_PCGRAD_ENC`` *and* the arm's own flag itself: relying on the caller to have left its
    patch in place silently compared an enabled arm against a disabled parent.

    Returns the arm's (delta-off) encoder gradients so the caller can assert that they still
    differ from the parent's fully-enabled reference.
    """
    restore_parent = patched(T_R4_08, "R4_08_PCGRAD_ENC", False)
    restore_arm = patched(T, flag, False)
    try:
        actual = run_step(build(T, seed), project=True, **kwargs)[0]
        expected = run_step(build(T_R4_08, seed), project=True)[0]
    finally:
        restore_arm()
        restore_parent()
    diff = max_abs_diff(actual, expected)
    assert diff < 1e-5, (
        f"{arm} with {flag} disabled must reproduce the PCGrad-free assembly, off by {diff}")
    parent_on = run_step(build(T_R4_08, seed), project=True)[0]
    assert max_abs_diff(actual, parent_on) > 1e-5, (
        f"{arm} with {flag} disabled must still differ from R4-08 ON, otherwise the arm's "
        "delta had no observable effect at all")
    return actual


# ----------------------------------------------------------------------------------- R5-03
def test_r5_03_quant_reference_differs_from_base_reference():
    T = MODULES["r5_03"]
    seed = SEED_PER_ARM["r5_03"]
    ref = build(T, seed)
    g_base, g_l1, g_other = reference_grads(ref)
    g_recon = T._r4_08_global_average(
        T._r4_08_encoder_gradients(ref["recon_loss"], ref["encoder"], retain=True))
    g_ref = T._r4_08_global_average(
        T._r4_08_encoder_gradients(ref["base_loss"] - ref["recon_loss"], ref["encoder"],
                                   retain=True))
    complement = max_abs_diff([a - b for a, b in zip(g_base, g_recon)], g_ref)
    assert complement < 1e-5, (
        f"g_base must equal g_recon + g_quant (quant reference is the complement), off by "
        f"{complement}")
    separation = float(flat(g_ref) @ flat(g_base)) / (float(flat(g_ref).norm())
                                                       * float(flat(g_base).norm()) + 1e-12)
    assert l2(g_ref, g_base) > 1e-5, "the quant-only reference must not coincide with g_base"
    print(f"  r5_03: cos(g_quant, g_base) = {separation:+.6f}", flush=True)

    expected = assemble(ref, g_base, unflatten_like(
        T.r4_project_conflict(flat(g_l1), flat(g_ref)), g_l1), g_other)
    case = build(T, seed)
    actual = run_step(case, project=True)[0]
    assert max_abs_diff(actual, expected) < 1e-5, (
        "r5_03 must project g_L1 against the quant-only gradient")
    assert_differentiated("r5_03", actual, reference_step(seed))


def test_r5_03_disabled_parity_with_r4_08():
    assert_pcgrad_free_parity("r5_03", MODULES["r5_03"], "R5_03_QUANT_REF",
                              SEED_PER_ARM["r5_03"])


# ----------------------------------------------------------------------------------- R5-04
def linear_blocks(T, case):
    """Per-Linear blocks, or the parent's single whole-vector block when the helper is absent."""
    if hasattr(T, "_r5_04_linear_blocks"):
        return T._r5_04_linear_blocks(case["model"].encoder, case["encoder"])
    return [list(range(len(case["encoder"])))]      # the parent has exactly one global block


def _block_dots(T, case, g_base, g_l1):
    blocks = linear_blocks(T, case)
    dots = []
    for block in blocks:
        b = torch.cat([g_base[i].reshape(-1) for i in block])
        l = torch.cat([g_l1[i].reshape(-1) for i in block])
        dots.append(float(l @ b))
    return blocks, dots


def test_r5_04_blockwise_projection_is_not_global_projection():
    T = MODULES["r5_04"]
    chosen = None
    for seed in range(20):
        ref = build(T, seed)
        g_base, g_l1, _ = reference_grads(ref)
        global_dot = float(flat(g_l1) @ flat(g_base))
        blocks, dots = _block_dots(T, ref, g_base, g_l1)
        if global_dot >= 0.0 and any(d < 0.0 for d in dots):
            chosen = (seed, global_dot, dots, blocks, g_base, g_l1, ref)
            break
    assert chosen is not None, (
        "no seed in 0..19 yields 'global dot >= 0 but some block dot < 0'; the blockwise "
        "counterexample cannot be constructed from real gradients in this range")
    seed, global_dot, dots, blocks, g_base, g_l1, ref = chosen
    _, g_other = reference_grads(ref)[0], reference_grads(ref)[2]
    print(f"  r5_04: seed {seed}, global dot {global_dot:+.4e} (no global projection) but "
          f"{sum(1 for d in dots if d < 0)}/{len(dots)} blocks conflict", flush=True)

    expected = [g.clone() for g in g_l1]
    for block in blocks:
        b = torch.cat([g_base[i].reshape(-1) for i in block])
        l = torch.cat([g_l1[i].reshape(-1) for i in block])
        projected = T.r4_project_conflict(l, b)
        offset = 0
        for i in block:
            count = g_l1[i].numel()
            expected[i] = projected[offset:offset + count].view_as(g_l1[i])
            offset += count
    case = build(T, seed)
    actual = run_step(case, project=True)[0]
    target = assemble(ref, g_base, expected, g_other)
    assert max_abs_diff(actual, target) < 1e-5, (
        "r5_04 must project independently inside each Linear weight+bias block")

    parent = reference_step(seed)
    assert l2(actual, parent) > 1e-5, "blockwise must differ from the global R4-08 projection"
    parent_ref = build(T_R4_08, seed)
    _, p_l1, _ = reference_grads(parent_ref)
    untouched = assemble(parent_ref, g_base, p_l1, g_other)
    changed = max_abs_diff(actual, untouched)
    assert changed > 1e-5, (
        "with a non-conflicting global dot R4-08 leaves the encoder untouched; r5_04 must not")


def test_r5_04_disabled_parity_with_r4_08():
    assert_pcgrad_free_parity("r5_04", MODULES["r5_04"], "R5_04_BLOCKWISE",
                              SEED_PER_ARM["r5_04"])


def test_r5_04_blocks_partition_every_linear_parameter():
    T = MODULES["r5_04"]
    case = build(T, SEED_PER_ARM["r5_04"])
    blocks = linear_blocks(T, case)
    seen = [i for block in blocks for i in block]
    assert sorted(seen) == list(range(len(case["encoder"]))), (
        "every encoder parameter must belong to exactly one block")
    assert len(blocks) >= 2, f"expected several blocks, got {len(blocks)}"
    again = T._r5_04_linear_blocks(case["model"].encoder, case["encoder"])
    assert blocks == again, "block partition must be deterministic for the same encoder"


# ----------------------------------------------------------------------------------- R5-05
def test_r5_05_half_strength_coefficient():
    T = MODULES["r5_05"]
    seed = SEED_PER_ARM["r5_05"]
    assert arm_attr(T, "R5_05_ALPHA", 1.0) == 0.5, (
        "r5_05 must use the module's alpha constant; the parent tree has no alpha "
        "override, so its effective alpha is 1.0")
    ref = build(T, seed)
    g_base, g_l1, g_other = reference_grads(ref)
    dot = float(flat(g_l1) @ flat(g_base))
    assert dot < 0.0, "fixture must produce a genuine conflict"
    base_sq = float(flat(g_base).pow(2).sum())
    hand = flat(g_l1) - T.R5_05_ALPHA * (dot / (base_sq + T.R4_08_EPS)) * flat(g_base)
    helper = T.r4_project_conflict(flat(g_l1), flat(g_base))
    assert float((helper - hand).abs().max()) < 1e-6, (
        "r5_05 must subtract exactly alpha * dot/(||g_base||^2+eps) * g_base")
    assert l2(helper, flat(g_l1) - (dot / (base_sq + T.R4_08_EPS)) * flat(g_base)) > 1e-6, (
        "alpha=0.5 must differ from the parent's alpha=1.0")

    expected = assemble(ref, g_base, unflatten_like(helper, g_l1), g_other)
    case = build(T, seed)
    actual = run_step(case, project=True)[0]
    assert max_abs_diff(actual, expected) < 1e-5, (
        "the executed r5_05 encoder gradient must match the half-strength assembly")
    assert_differentiated("r5_05", actual, reference_step(seed))


def test_r5_05_disabled_parity_with_r4_08():
    T = MODULES["r5_05"]
    seed = SEED_PER_ARM["r5_05"]
    restore = patched(T, "R5_05_ALPHA", 1.0)
    try:
        actual = run_step(build(T, seed), project=True)[0]
    finally:
        restore()
    parent = run_step(build(T_R4_08, seed), project=True)[0]
    assert max_abs_diff(actual, parent) < 1e-5, (
        "alpha=1.0 must reproduce the parent's full-strength projection")


# ----------------------------------------------------------------------------------- R5-06
def test_r5_06_restores_original_l1_norm():
    T = MODULES["r5_06"]
    seed = SEED_PER_ARM["r5_06"]
    ref = build(T, seed)
    g_base, g_l1, g_other = reference_grads(ref)
    projected = unflatten_like(T.r4_project_conflict(flat(g_l1), flat(g_base)), g_l1)
    expected = assemble(ref, g_base, projected, g_other)
    case = build(T, seed)
    actual = run_step(case, project=True)[0]

    recovered = [(actual[i] - g_base[i]) / case["scale"] - g_other[i]
                 for i in range(len(actual))]
    original_norm = float(flat(g_l1).norm())
    recovered_norm = float(flat(recovered).norm())
    assert abs(recovered_norm - original_norm) <= 1e-5 * max(original_norm, 1e-12), (
        f"r5_06 must restore ||g_L1||: got {recovered_norm}, expected {original_norm}")
    cosine = float(flat(recovered) @ flat(projected)) / (
        recovered_norm * float(flat(projected).norm()) + 1e-12)
    assert cosine > 1.0 - 1e-5, (
        f"the restored vector must stay parallel to the parent projection, cos={cosine}")
    parent = build(T_R4_08, seed)
    _, p_l1, p_other = reference_grads(parent)
    p_base = reference_grads(parent)[0]
    p_proj = unflatten_like(T_R4_08.r4_project_conflict(flat(p_l1), flat(p_base)), p_l1)
    parent_norm = float(flat(p_proj).norm())
    assert abs(parent_norm - original_norm) > 1e-5 * original_norm, (
        "R4-08 must actually shrink ||g_L1||, otherwise the restoration is a no-op")
    print(f"  r5_06: ||g_L1|| {original_norm:.6f} -> restored {recovered_norm:.6f} "
          f"(R4-08 would give {parent_norm:.6f})", flush=True)


def test_r5_06_degenerate_projection_returns_zeros():
    T = MODULES["r5_06"]
    g_l1 = torch.tensor([1.0, 0.0])
    g_base = torch.tensor([-2.0, 0.0])         # anti-parallel: the projection annihilates g_L1
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    assert torch.equal(out, torch.zeros_like(out)), (
        "a projection that annihilates g_L1 must return exact zeros, never NaN")


def test_r5_06_disabled_parity_with_r4_08():
    T = MODULES["r5_06"]
    seed = SEED_PER_ARM["r5_06"]
    restore = patched(T, "R5_06_NORM_PRESERVE", False)
    try:
        actual = run_step(build(T, seed), project=True)[0]
    finally:
        restore()
    parent = run_step(build(T_R4_08, seed), project=True)[0]
    assert max_abs_diff(actual, parent) < 1e-5, (
        "norm restoration off must reproduce the parent's projection exactly")


# ----------------------------------------------------------------------------------- R5-07
def _rotate_to_cos(g_l1: torch.Tensor, g_base: torch.Tensor, target: float):
    """Rotate g_base toward -g_L1 until ``cos(g_L1, g_base) == target`` (bisection)."""
    goal = -flat(g_l1) / (flat(g_l1).norm() + 1e-12) * float(flat(g_base).norm())
    lo, hi = 0.0, 1.0

    def cos_at(t):
        candidate = (1.0 - t) * flat(g_base) + t * goal
        return float(flat(g_l1) @ candidate) / (
            float(flat(g_l1).norm()) * float(candidate.norm()) + 1e-12)

    if cos_at(hi) > target:
        return None, cos_at(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cos_at(mid) > target:
            lo = mid
        else:
            hi = mid
    return (1.0 - hi) * flat(g_base) + hi * goal, cos_at(hi)


def test_r5_07_cosine_cutoff_boundary():
    T = MODULES["r5_07"]
    assert arm_attr(T, "R5_07_COS_THRESHOLD", float("-inf")) == -0.2, (
        "r5_07 must gate on cos < -0.2; the parent tree has no cosine gate at all, "
        "so it projects on every negative dot including cos == -0.2")
    seed = SEED_PER_ARM["r5_07"]
    ref = build(T, seed)
    g_base, g_l1, _ = reference_grads(ref)

    weak, cos_weak = _rotate_to_cos(flat(g_l1), flat(g_base), -0.1)
    assert weak is not None and cos_weak > -0.2, "need a non-conflicting rotation"
    before = T.R4_08_PROJECTION_TRACE["active_count"]
    out = T.r4_project_conflict(flat(g_l1), weak)
    assert T.R4_08_PROJECTION_TRACE["active_count"] == before, (
        f"cos={cos_weak:.4f} >= -0.2 must not project")
    assert torch.equal(out, flat(g_l1)), "cos >= -0.2 must pass g_L1 through bitwise"

    strong, cos_strong = _rotate_to_cos(flat(g_l1), flat(g_base), -0.5)
    assert strong is not None and cos_strong < -0.2, "need a strongly conflicting rotation"
    before = T.R4_08_PROJECTION_TRACE["active_count"]
    out = T.r4_project_conflict(flat(g_l1), strong)
    assert T.R4_08_PROJECTION_TRACE["active_count"] == before + 1, (
        f"cos={cos_strong:.4f} < -0.2 must project exactly once")
    hand = flat(g_l1) - (float(flat(g_l1) @ strong)
                         / (float(strong.pow(2).sum()) + T.R4_08_EPS)) * strong
    assert float((out - hand).abs().max()) < 1e-6, (
        "the strong-conflict branch must apply the parent's full projection")

    boundary, cos_boundary = _rotate_to_cos(flat(g_l1), flat(g_base), -0.2)
    # float32 distances cap the achievable cos resolution; 1e-6 is the tightest reachable
    assert boundary is not None and abs(cos_boundary + 0.2) < 1e-6, (
        f"bisection must reach the boundary, reached cos={cos_boundary}")
    before = T.R4_08_PROJECTION_TRACE["active_count"]
    T.r4_project_conflict(flat(g_l1), boundary)
    assert T.R4_08_PROJECTION_TRACE["active_count"] == before, (
        "cos == -0.2 exactly must NOT project (the comparison is strict)")
    print(f"  r5_07: boundary reached at cos={cos_boundary:.12f} (no projection), "
          f"strong branch at cos={cos_strong:.4f}", flush=True)


def test_r5_07_disabled_parity_with_r4_08():
    T = MODULES["r5_07"]
    seed = SEED_PER_ARM["r5_07"]
    # `if cos >= threshold: return g_l1`, so a threshold ABOVE every achievable cosine (cos <= 1)
    # always takes the projection branch and reproduces the parent. A threshold of -1.0 would do
    # the opposite: cos >= -1 always holds, so nothing would ever project.
    restore = patched(T, "R5_07_COS_THRESHOLD", 1.5)
    try:
        actual = run_step(build(T, seed), project=True)[0]
    finally:
        restore()
    parent = run_step(build(T_R4_08, seed), project=True)[0]
    assert max_abs_diff(actual, parent) < 1e-5, (
        "threshold 1.5 forces the projection branch and must equal R4-08")


# ----------------------------------------------------------------------------------- R5-08
def test_r5_08_epoch_2249_2250_boundary():
    T = MODULES["r5_08"]
    assert arm_attr(T, "R5_08_LATE_START_EPOCH", 0) == 2250, (
        "r5_08 must gate on epoch >= 2250; the parent tree has no epoch gate, so it "
        "projects at every epoch")
    seed = SEED_PER_ARM["r5_08"]
    ref = build(T, seed)
    g_base, g_l1, g_other = reference_grads(ref)
    unprojected = assemble(ref, g_base, g_l1, g_other)
    projected = assemble(ref, g_base, unflatten_like(
        T.r4_project_conflict(flat(g_l1), flat(g_base)), g_l1), g_other)

    early = run_step(build(T, seed), project=True, epoch=2249)[0]
    late = run_step(build(T, seed), project=True, epoch=2250)[0]
    assert max_abs_diff(early, unprojected) < 1e-5, (
        "epoch 2249 must take the plain unprojected path")
    assert max_abs_diff(late, projected) < 1e-5, (
        "epoch 2250 must take the projected path")
    assert l2(early, late) > 1e-5, "the boundary must actually change the encoder gradient"
    print(f"  r5_08: ||grad(2249) - grad(2250)||_2 = {l2(early, late):.6e}", flush=True)


def test_r5_08_disabled_parity_with_r4_08():
    T = MODULES["r5_08"]
    seed = SEED_PER_ARM["r5_08"]
    restore = patched(T, "R5_08_LATE_START_EPOCH", 0)
    try:
        actual = run_step(build(T, seed), project=True, epoch=2250)[0]
    finally:
        restore()
    parent = run_step(build(T_R4_08, seed), project=True)[0]
    assert max_abs_diff(actual, parent) < 1e-5, (
        "with the epoch gate open, r5_08 must equal R4-08 at every epoch")


# ----------------------------------------------------------------------------------- R5-09
def test_r5_09_two_step_ema_state():
    """R5-09 must project against a LAGGING reference, not against the current g_base.

    The first, purely behavioural claim is what makes this a valid RED: on the parent tree (no
    EMA at all) the second step reproduces the plain R4-08 projection of the second seed, so the
    two are bitwise equal and the assertion fails on a numeric comparison rather than on a
    missing attribute. Only once the EMA is known to exist do we check the buffer arithmetic.
    """
    T = MODULES["r5_09"]
    seed1, seed2 = CONFLICT_SEEDS[0], CONFLICT_SEEDS[1]
    restore_ema = reset_ema(T)
    try:
        run_step(build(T, seed1), project=True)
        second = build(T, seed2)
        g_base2, g_l1_2, g_other_2 = reference_grads(second)
        actual = run_step(second, project=True)[0]
        reference = reference_step(seed2)
        separation = l2(actual, reference)
        assert separation > 1e-5, (
            f"after a first step the R5-09 reference must lag g_base, so the second step must "
            f"differ from the plain R4-08 projection of that same seed; measured {separation:.3e}. "
            "A value of 0 means the reference is not an EMA at all.")
        assert max_abs_diff(actual, reference_step(seed1)) > 1e-5, (
            "step 2 must not simply repeat step 1's projection")

        state = ema_state(T)
        if state is None or state["buffers"] is None:
            raise AssertionError(
                "R5-09 must maintain an EMA buffer; none is present after two steps")
        buffers2 = [b.detach().clone() for b in state["buffers"]]
        decay = arm_attr(T, "R5_09_EMA_DECAY", 1.0)
        g_base1 = reference_grads(build(T, seed1))[0]
        hand = [decay * a + (1.0 - decay) * b for a, b in zip(g_base1, g_base2)]
        update_err = max_abs_diff(buffers2, hand)
        assert update_err < 1e-7, (
            f"after two steps the buffer must equal 0.9*m1 + 0.1*g_base_2, off by {update_err}")
        # `_r4_08_apply_projection` calls `_r5_09_update_ema(g_base)` and projects against the
        # value it RETURNS, i.e. this step's own post-update buffer m2 -- not the buffer as it
        # stood before the step, and certainly not m3.
        expected = assemble(second, g_base2, unflatten_like(
            T.r4_project_conflict(flat(g_l1_2), flat(hand)), g_l1_2), g_other_2)
        against_raw = assemble(second, g_base2, unflatten_like(
            T.r4_project_conflict(flat(g_l1_2), flat(g_base2)), g_l1_2), g_other_2)
        consumed_err = max_abs_diff(actual, expected)
        assert consumed_err < 1e-5, (
            f"the executed encoder gradient must consume the EMA buffer m2, off by {consumed_err}")
        assert max_abs_diff(expected, against_raw) > 1e-6, (
            "m2 and the raw g_base must give different projections, else the lag is untested")
        print(f"  r5_09: step-2 vs R4-08 {separation:.6e}, buffer update err {update_err:.3e}, "
              f"consumed-m2 err {consumed_err:.3e}", flush=True)
    finally:
        restore_ema()


def test_r5_09_reference_is_not_current_base():
    """Purely behavioural: the reference used on step 2 must not be that step's own g_base."""
    T = MODULES["r5_09"]
    seed1, seed2 = CONFLICT_SEEDS[0], CONFLICT_SEEDS[1]
    restore_ema = reset_ema(T)
    try:
        run_step(build(T, seed1), project=True)
        second = build(T, seed2)
        g_base2 = reference_grads(second)[0]
        actual = run_step(second, project=True)[0]
        fresh = run_step(build(T, seed2), project=True, skip_projection=False)[0]
        separation = l2(actual, reference_step(seed2))
        assert separation > 1e-5, (
            "the EMA reference must lag g_base, so step 2 must differ from the plain projection "
            f"of that same seed; measured {separation:.3e}")
        assert max_abs_diff(actual, fresh) > 1e-5, (
            "the reference must carry state across steps, otherwise the two are independent")
        print(f"  r5_09: step-2 executed vs plain projection {separation:.6e} "
              f"(reference genuinely lags g_base)", flush=True)
    finally:
        restore_ema()


def test_r5_09_differentiated_from_r4_08():
    """R5-09 is only differentiable from step 2 onwards; step 1 equals R4-08 exactly."""
    T = MODULES["r5_09"]
    seed1, seed2 = CONFLICT_SEEDS[0], CONFLICT_SEEDS[1]
    restore_ema = reset_ema(T)
    try:
        first_actual = run_step(build(T, seed1), project=True)[0]
        second_actual = run_step(build(T, seed2), project=True)[0]
    finally:
        restore_ema()
    first_ref = reference_step(seed1)
    assert max_abs_diff(first_actual, first_ref) < 1e-5, (
        "step 1 seeds the EMA with g_base, so it must equal R4-08 exactly")
    assert_differentiated("r5_09 @step2", second_actual, reference_step(seed2))


def test_r5_09_disabled_parity_with_r4_08():
    T = MODULES["r5_09"]
    seed = SEED_PER_ARM["r5_09"]
    restore_alpha = patched(T, "R5_09_EMA_DECAY", 1.0)
    restore_ema = reset_ema(T)
    try:
        actual = run_step(build(T, seed), project=True)[0]
    finally:
        restore_ema()
        restore_alpha()
    parent = run_step(build(T_R4_08, seed), project=True)[0]
    assert max_abs_diff(actual, parent) < 1e-5, (
        "decay=1.0 with a freshly seeded buffer makes the reference equal g_base, i.e. R4-08")


# ---------------------------------------------------------------------------------- R5-10
def test_r5_10_projects_against_true_l2_l3_reference():
    """AUDIT FINDING: the "true L2+L3 reference" is structurally ZERO, so R5-10 is a no-op.

    ``VQLayer.forward`` uses the straight-through estimator ``x_q = x + (x_q - x).detach()``, so
    ``d(x_q)/dx == 1`` and, because ``residual_k = residual_{k-1} - x_q_{k-1}``,
    ``d(residual_k)/d(residual_{k-1}) == 1 - 1 == 0``. Levels L2 and L3 therefore contribute an
    identically zero ENCODER gradient, ``g_other == 0``, even though ``other_loss`` itself is a
    large non-zero number. ``r4_project_conflict(g_L1, 0)`` hits the ``base_sq <= 0`` passthrough,
    so R5-10 never projects and its assembly is the PCGrad-free ``g_base + (w/v)*g_L1`` --
    never the parent's projection.

    This test pins that structural fact rather than pretending R5-10 has an effect.
    """
    T = MODULES["r5_10"]
    seed = SEED_PER_ARM["r5_10"]
    zeros_seen = []
    for probe_seed in CONFLICT_SEEDS:
        case = build(T, probe_seed)
        assert float(case["other_loss"].detach()) > 0.0, (
            "the L2+L3 loss value must be non-trivial for this finding to mean anything")
        g_base, g_l1, g_other = reference_grads(case)
        zeros_seen.append(float(flat(g_other).norm()))
        raw = T._r4_08_encoder_gradients(case["other_loss"], case["encoder"], retain=True)
        assert all(g is not None for g in raw), (
            "the L2+L3 term must be connected to the encoder graph (not merely unused), "
            "otherwise this test would be measuring the wrong thing")
    assert max(zeros_seen) == 0.0, (
        f"the encoder gradient of the L2+L3 term must be exactly zero, saw {zeros_seen}")

    # with a zero reference the projection is a passthrough -> the assembly is PCGrad-free
    ref = build(T, seed)
    g_base, g_l1, g_other = reference_grads(ref)
    passthrough = T.r4_project_conflict(flat(g_l1), flat(g_other))
    assert torch.equal(passthrough, flat(g_l1)), (
        "projecting against a zero reference must pass g_L1 through unchanged")
    actual = run_step(build(T, seed), project=True)[0]
    expected = assemble(ref, g_base, unflatten_like(passthrough, g_l1), g_other)
    assert max_abs_diff(actual, expected) < 1e-5, (
        "r5_10's executed encoder gradient must be the PCGrad-free assembly")

    parent = reference_step(seed)
    assert l2(actual, parent) > 1e-5, (
        "R5-10 must differ from R4-08 -- but only by NOT projecting at all")
    _, parent_l1, _ = reference_grads(build(T_R4_08, seed))
    parent_base = reference_grads(build(T_R4_08, seed))[0]
    parent_assembly = assemble(build(T_R4_08, seed), parent_base,
                               unflatten_like(T_R4_08.r4_project_conflict(
                                   flat(parent_l1), flat(parent_base)), parent_l1),
                               reference_grads(build(T_R4_08, seed))[2])
    assert max_abs_diff(actual, parent_assembly) > 1e-5, (
        "R5-10 must NOT reproduce the parent's projection: its mechanism is inert")
    print(f"  r5_10: ||g_other|| == 0 for {len(zeros_seen)}/{len(zeros_seen)} seeds; "
          f"R5-10 therefore runs the PCGrad-free assembly, not the R4-08 projection", flush=True)


def test_r5_10_absent_other_is_passthrough():
    T = MODULES["r5_10"]
    seed = SEED_PER_ARM["r5_10"]
    ref = build(T, seed)
    g_base, g_l1, _ = reference_grads(ref)
    expected = [g_base[i] + ref["scale"] * g_l1[i] for i in range(len(g_base))]

    case = build(T, seed)
    case["other_loss"] = None
    before = T.R4_08_PROJECTION_TRACE["active_count"]
    actual = run_step(case, project=True)[0]
    assert T.R4_08_PROJECTION_TRACE["active_count"] == before, (
        "with no L2+L3 term the inter-level projection must be skipped entirely")
    assert max_abs_diff(actual, expected) < 1e-5, (
        "without the reference gradient the assembly must be g_base + scale*g_L1, unprojected")


def test_r5_10_disabled_parity_with_r4_08():
    assert_pcgrad_free_parity("r5_10", MODULES["r5_10"], "R5_10_INTERLEVEL",
                              SEED_PER_ARM["r5_10"])


# ------------------------------------------------------------------------- global invariants
def test_no_unexpected_projection_active():
    """No arm may fire more projections than its own rule allows.

    R5-04 projects once per Linear weight+bias block, so its allowance is the block count;
    every other arm performs a single global projection per step.
    """
    for arm in ARMS:
        T = MODULES[arm]
        seed = SEED_PER_ARM[arm]
        case = build(T, seed)
        allowance = len(linear_blocks(T, case))
        restore_ema = reset_ema(T) if arm == "r5_09" else (lambda: None)
        try:
            before = T.R4_08_PROJECTION_TRACE["active_count"]
            run_step(case, project=True)
            fired = T.R4_08_PROJECTION_TRACE["active_count"] - before
        finally:
            restore_ema()
        assert fired <= allowance, (
            f"{arm} fired {fired} projections in one step; its rule allows at most {allowance}")
        print(f"  {arm}: projections fired this step = {fired} (allowance {allowance})", flush=True)


def test_no_extra_optimizer_steps():
    """Exactly one AdamW step per call."""
    for arm in ("r5_03", "r5_06", "r5_10"):
        T = MODULES[arm]
        seed = SEED_PER_ARM[arm]
        _, optimizer = run_step(build(T, seed), project=True)
        steps = {int(state["step"].item()) for state in optimizer.state.values()}
        assert steps == {1}, f"{arm}: one call must advance every AdamW state exactly once, got {steps}"


def main() -> int:
    print(f"seeds per arm: {SEED_PER_ARM}", flush=True)
    tests = [
        test_r5_03_quant_reference_differs_from_base_reference,
        test_r5_03_disabled_parity_with_r4_08,
        test_r5_04_blockwise_projection_is_not_global_projection,
        test_r5_04_disabled_parity_with_r4_08,
        test_r5_04_blocks_partition_every_linear_parameter,
        test_r5_05_half_strength_coefficient,
        test_r5_05_disabled_parity_with_r4_08,
        test_r5_06_restores_original_l1_norm,
        test_r5_06_degenerate_projection_returns_zeros,
        test_r5_06_disabled_parity_with_r4_08,
        test_r5_07_cosine_cutoff_boundary,
        test_r5_07_disabled_parity_with_r4_08,
        test_r5_08_epoch_2249_2250_boundary,
        test_r5_08_disabled_parity_with_r4_08,
        test_r5_09_two_step_ema_state,
        test_r5_09_reference_is_not_current_base,
        test_r5_09_differentiated_from_r4_08,
        test_r5_09_disabled_parity_with_r4_08,
        test_r5_10_projects_against_true_l2_l3_reference,
        test_r5_10_absent_other_is_passthrough,
        test_r5_10_disabled_parity_with_r4_08,
        test_no_unexpected_projection_active,
        test_no_extra_optimizer_steps,
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
        print("COUNTEREXAMPLE_TESTS_RED", flush=True)
        return 1
    print("ALL TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
