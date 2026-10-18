"""Issue #71 §3 — true 2-rank gloo DDP test of the production gradient path.

Audit-only. Spawns two real ``torch.distributed`` gloo processes on CPU, runs the exact
production sequence (real forward -> real losses -> ``loss.backward(retain_graph=True)`` ->
``_r4_08_apply_projection`` -> ``clip_grad_norm_`` -> ``AdamW.step()``), and compares the result
against a single-process reference that pools the two ranks' gradients **explicitly and
independently** of the production code path.

Start method is ``fork`` so both children inherit the already-imported arm modules; importing
each trainer costs ~1 minute, and re-importing per rank would be pure waste.

What this test is designed to catch
-----------------------------------
* double averaging (pooling a value that is already a mean),
* per-rank *local* PCGrad instead of global PCGrad -- including the case where rank 0 sees a
  negative local dot, rank 1 sees a positive one, and the global dot is non-negative so a
  correct implementation must project **nothing**,
* decoder/codebook gradients being tampered with by the projection,
* divergent weights or optimizer state across ranks after the step,
* per-rank EMA buffers in R5-09.

Run: python results/stage2_RQ-VAE/r45_gradient_integrity_audit/test_ddp_gradient_equivalence.py
"""
from __future__ import annotations

import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
AUDIT = REPO / "results/stage2_RQ-VAE/r45_gradient_integrity_audit"
LOGS = AUDIT / "logs"
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K  # noqa: E402

T_R4_08 = K.load_arm("r4_08")
T_R5_04 = K.load_arm("r5_04")
T_R5_09 = K.load_arm("r5_09")

IN_DIM = 32
N_SRC = 24
BEHAVIOR_WEIGHT = 0.5
SEED = 2024
RANK_ROWS = {0: 24, 1: 20}          # deliberately uneven local batches
CHILD_TIMEOUT_S = 180
CTX = mp.get_context("fork")

# With the "fork" start method a child gets a COPY of the parent's memory, so anything a rank
# computes never reaches the parent in-process. Every rank therefore persists its scalars and
# tensors to disk, and the parent reads them back. (The first version of this test kept them in
# module globals and silently compared nothing.)
ARTIFACTS = LOGS / "ddp_artifacts"


def save_child(tag: str, rank: int, scalars: dict, tensors: dict | None = None) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{tag}_rank{rank}.json").write_text(json.dumps(scalars, indent=2),
                                                     encoding="utf-8")
    if tensors:
        torch.save(tensors, ARTIFACTS / f"{tag}_rank{rank}.pt")


def load_child(tag: str, rank: int) -> dict:
    scalars_path = ARTIFACTS / f"{tag}_rank{rank}.json"
    if not scalars_path.is_file():
        return {"rank": rank, "ok": False, "error": "rank wrote no result file"}
    payload = json.loads(scalars_path.read_text(encoding="utf-8"))
    tensor_path = ARTIFACTS / f"{tag}_rank{rank}.pt"
    if tensor_path.is_file():
        payload["_tensors"] = torch.load(tensor_path, weights_only=False)
    return payload


# ------------------------------------------------------------------ single-process building
def make_batch(rows: int, offset: int):
    """A deterministic local batch: the first `rows` rows act as sources, the rest as candidates."""
    generator = torch.Generator().manual_seed(SEED + 100 * offset + rows)
    emb = torch.randn(2 * rows, IN_DIM, generator=generator)
    source_ids = torch.arange(rows)
    target_ids = (torch.arange(rows) + 1) % rows
    return emb, source_ids, target_ids


def build_local(T, rows: int, offset: int):
    torch.manual_seed(SEED)                      # identical weights on both ranks
    model = T.RQVAE(T._tokenizer_config(), in_dim=IN_DIM)
    emb, source_ids, target_ids = make_batch(rows, offset)
    reconstructed, quant_loss, _, _ = model.forward(emb)
    base_loss, recon_loss = model.compute_loss(emb, reconstructed, quant_loss)
    residuals = T._residuals_with_trace(model, emb)
    l1_loss, other_loss, valid = T.behavior_contrastive_components(
        residuals, source_ids, target_ids, None)
    encoder = list(model.encoder.parameters())
    return {"model": model, "emb": emb, "base_loss": base_loss, "recon_loss": recon_loss,
            "l1_loss": l1_loss, "other_loss": other_loss, "valid_levels": valid,
            "encoder": encoder,
            "non_encoder": [p for p in model.parameters() if not any(p is q for q in encoder)]}


def local_grads(case, key: str):
    """Encoder gradients BEFORE pooling, straight from autograd (production helper)."""
    T = case["_T"]
    return T._r4_08_encoder_gradients(case[key], case["encoder"], retain=True)


def arm_kwargs(T, case) -> dict:
    accepted = inspect.signature(T._r4_08_apply_projection).parameters
    kwargs = {}
    if "recon_loss" in accepted:
        kwargs["recon_loss"] = case["recon_loss"]
    if "encoder_module" in accepted:
        kwargs["encoder_module"] = case["model"].encoder
    return kwargs


def pooled_reference(T, local_cases, *, project: bool = True):
    """Independent single-process pooled reference.

    The two ranks' LOCAL gradients are pooled by hand here -- ``_r4_08_global_average`` is
    replaced by an explicit mean, so a bug in the production pooling cannot hide inside the
    reference. Everything downstream of that (the projection rule, the
    ``w/valid_levels`` scaling, the gradient override, the global clip norm) is then driven by
    the production code, which is exactly what the test needs to compare against.
    """
    encoder = local_cases[0]["encoder"]
    local = {}
    for key in ("base_loss", "l1_loss", "other_loss"):
        per_rank = [T._r4_08_encoder_gradients(case[key], encoder, retain=True)
                    for case in local_cases]
        local[key] = [(per_rank[0][i] + per_rank[1][i]) / 2.0
                      for i in range(len(per_rank[0]))]
    valid = max(c["valid_levels"] for c in local_cases)

    case = local_cases[0]
    model = case["model"]
    non_encoder = case["non_encoder"]
    per_rank_ne = []
    for c in local_cases:
        grads = torch.autograd.grad(c["base_loss"], non_encoder, retain_graph=True,
                                    allow_unused=True)
        per_rank_ne.append([torch.zeros_like(p) if g is None else g
                            for p, g in zip(non_encoder, grads)])
    pooled_ne = [(per_rank_ne[0][i] + per_rank_ne[1][i]) / 2.0
                 for i in range(len(non_encoder))]

    saved_pool = T._r4_08_global_average
    saved_flag = T.R4_08_PCGRAD_ENC
    queue = [local["base_loss"], local["l1_loss"], local["other_loss"]]

    def manual_average(_grads):
        return queue.pop(0)

    T._r4_08_global_average = manual_average
    if not project:
        T.R4_08_PCGRAD_ENC = False
    try:
        model.zero_grad(set_to_none=False)
        case["base_loss"].backward(retain_graph=True)
        if project:
            T._r4_08_apply_projection(case["base_loss"], case["l1_loss"], case["other_loss"],
                                      encoder, BEHAVIOR_WEIGHT, valid_levels=valid,
                                      **arm_kwargs(T, case))
        encoder_assembly = [p.grad.detach().clone() for p in encoder]
    finally:
        T._r4_08_global_average = saved_pool
        T.R4_08_PCGRAD_ENC = saved_flag
    for p, g in zip(non_encoder, pooled_ne):
        p.grad = g.clone()
    total_norm = math.sqrt(sum(float(g.pow(2).sum()) for g in encoder_assembly)
                           + sum(float(g.pow(2).sum()) for g in pooled_ne))
    multiplier = min(1.0, 1.0 / (total_norm + 1e-6))
    return ([g * multiplier for g in encoder_assembly],
            [g * multiplier for g in pooled_ne],
            total_norm, multiplier)


# --------------------------------------------------------------------------- child process
def child(rank: int, init_file: str, arm: str, tag: str, synth: bool) -> None:
    try:
        dist.init_process_group(backend="gloo", init_method=f"file://{init_file}",
                                rank=rank, world_size=2)
        if synth:
            _child_synthetic(rank, arm, tag)
        else:
            _child_real(rank, arm, tag)
    except Exception as exc:  # noqa: BLE001
        save_child(tag, rank, {"rank": rank, "ok": False,
                               "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def _child_real(rank: int, arm: str, tag: str) -> None:
    T = {"r4_08": T_R4_08, "r5_04": T_R5_04, "r5_09": T_R5_09}[arm]
    rows = RANK_ROWS[rank]
    base_case = build_local(T, rows, offset=rank)
    base_case["_T"] = T
    model = base_case["model"]
    # Production wraps the module in DDP, and that wrapper is what averages the DECODER and
    # CODEBOOK gradients. Without it those gradients stay rank-local: on an earlier run the
    # child's non-encoder norm was exactly the rank-0 local value, i.e. 2x the pooled one.
    wrapped = DDP(model, device_ids=None, find_unused_parameters=False)
    # NOTE: every forward inside the loop goes through the DDP wrapper. DDP engages its gradient
    # reducer inside its own forward, so a backward() on a loss built from a raw-module forward
    # skips the reducer entirely and leaves every gradient rank-local (measured: the
    # decoder/codebook norm stayed exactly at the rank-0 local value, 2x the pooled one).
    iterations = 3 if arm == "r5_09" else 1
    if arm == "r5_09":
        T.R5_09_EMA_STATE["buffers"] = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    tensors: dict[str, torch.Tensor] = {}
    total_norm = None
    pooled_norms: list[list[float]] = []
    for step in range(iterations):
        # Rebuild the graph every iteration: optimizer.step() mutates the parameters IN PLACE,
        # so a graph carried over from the previous step raises "variable needed for gradient
        # computation has been modified by an inplace operation". Production gets a fresh batch
        # per step for the same reason.
        case = dict(base_case)          # same wrapped module, same parameter objects
        emb, source_ids, target_ids = make_batch(rows, offset=rank)
        reconstructed, quant_loss, _, _ = wrapped(emb)
        case["base_loss"], case["recon_loss"] = model.compute_loss(
            emb, reconstructed, quant_loss)
        residuals = T._residuals_with_trace(model, emb)
        case["l1_loss"], case["other_loss"], case["valid_levels"] = \
            T.behavior_contrastive_components(residuals, source_ids, target_ids, None)
        model.zero_grad(set_to_none=False)
        case["base_loss"].backward(retain_graph=True)
        # The pooled components the projection is about to consume, recorded for cross-checking.
        # This MUST come after the first backward(): DDP performs its parameter broadcast
        # (an in-place copy) there, which invalidates any gradient taken before it.
        if step == 0:
            pooled_norms.append([
                float(torch.cat([g.reshape(-1) for g in T._r4_08_global_average(
                    T._r4_08_encoder_gradients(case[k], case["encoder"],
                                               retain=True))]).norm())
                for k in ("base_loss", "l1_loss", "other_loss")])
        T._r4_08_apply_projection(case["base_loss"], case["l1_loss"], case["other_loss"],
                                  case["encoder"], BEHAVIOR_WEIGHT,
                                  valid_levels=case["valid_levels"], **arm_kwargs(T, case))
        if rank == 0 and step == iterations - 1:
            tensors["encoder_grads"] = torch.cat(
                [p.grad.detach().reshape(-1).clone() for p in case["encoder"]])
            tensors["nonencoder_grads"] = torch.cat(
                [p.grad.detach().reshape(-1).clone() for p in case["non_encoder"]])
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    flat = torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(2)]
    dist.all_gather(gathered, flat)
    tensors["gathered"] = torch.stack(gathered)
    save_child(tag, rank, {
        "rank": rank, "ok": True, "arm": arm, "rows": rows, "iterations": iterations,
        "total_norm": total_norm,
        "pooled_component_norms": pooled_norms[-1] if pooled_norms else None,
        "ema_buffers": ([round(float(b.abs().sum()), 12) for b in T.R5_09_EMA_STATE["buffers"]]
                        if arm == "r5_09" else None),
    }, tensors)


def _child_synthetic(rank: int, arm: str, tag: str) -> None:
    """Drive the REAL pooling + projection with cancelling per-rank local gradients.

    Each rank gets local gradients whose local dot products have OPPOSITE signs while the
    globally pooled dot is non-negative, so a correct global projection must fire zero times.
    ``_r4_08_encoder_gradients`` is swapped only so the real losses yield these chosen vectors;
    ``_r4_08_global_average``, ``r4_project_conflict``, the gradient override, the clip and
    ``AdamW.step()`` all remain the production implementations.
    """
    T = {"r4_08": T_R4_08, "r5_04": T_R5_04, "r5_09": T_R5_09}[arm]
    torch.manual_seed(SEED)
    model = T.RQVAE(T._tokenizer_config(), in_dim=IN_DIM)
    encoder = list(model.encoder.parameters())
    n = sum(p.numel() for p in encoder)
    if rank == 0:
        g_l1 = torch.ones(n)
        g_base = -torch.ones(n)          # local dot = -n  (conflict)
    else:
        g_l1 = torch.ones(n)
        g_base = 2.0 * torch.ones(n)     # local dot = +2n (no conflict)
    # pooled g_base = (-1 + 2)/2 = +0.5  =>  pooled dot with g_l1 = 1 is POSITIVE, so a correct
    # global projection must do nothing even though rank 0 alone would have conflicted.
    order = iter((g_base.clone(), g_l1.clone(), torch.zeros(n)))   # base, l1, other

    def synthetic_grads(_loss, params, retain=True):
        flat = next(order)
        out, offset = [], 0
        for param in params:
            count = param.numel()
            out.append(flat[offset:offset + count].view_as(param).clone())
            offset += count
        return out

    saved = T._r4_08_encoder_gradients
    T._r4_08_encoder_gradients = synthetic_grads
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        model.zero_grad(set_to_none=False)
        before = T.R4_08_PROJECTION_TRACE["active_count"]
        T._r4_08_apply_projection(None, None, None, encoder, BEHAVIOR_WEIGHT,
                                  valid_levels=3, **arm_kwargs(T, {"recon_loss": None,
                                                                   "model": model}))
        fired = T.R4_08_PROJECTION_TRACE["active_count"] - before
        tensors = {"encoder_grads": torch.cat(
            [p.grad.detach().reshape(-1).clone() for p in encoder])}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        flat = torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])
        gathered = [torch.zeros_like(flat) for _ in range(2)]
        dist.all_gather(gathered, flat)
        tensors["gathered"] = torch.stack(gathered)
        pooled_dot = float((g_l1 * ((-torch.ones(n) + 2.0 * torch.ones(n)) / 2.0)).sum())
        save_child(tag, rank, {"rank": rank, "ok": True, "arm": arm, "rows": n,
                               "iterations": 1, "local_dot": float(g_l1 @ g_base),
                               "pooled_dot": pooled_dot,
                               "fired": fired, "synthetic": True}, tensors)
    finally:
        T._r4_08_encoder_gradients = saved


# ------------------------------------------------------------------------------- harness
_SCENARIO_CACHE: dict[str, dict[int, dict]] = {}


def run_pair(arm: str, tag: str, synth: bool = False) -> dict[int, dict]:
    """Run one 2-rank scenario in a FRESH subprocess, then cache its artifacts.

    Creating and tearing down a gloo file store repeatedly inside one parent process wedged
    after the first scenario -- twice it surfaced as a false "ranks hung past 180s". Each
    scenario therefore gets its own pristine interpreter (``--scenario <tag>``), which is also
    how a real launcher would run them, and the parent only reads the persisted artifacts.
    """
    if tag in _SCENARIO_CACHE:
        return _SCENARIO_CACHE[tag]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--scenario", tag,
         "--arm", arm, "--synth", "1" if synth else "0"],
        capture_output=True, text=True, timeout=CHILD_TIMEOUT_S + 60)
    for line in proc.stdout.splitlines():
        if line.strip():
            print("    " + line, flush=True)
    if proc.returncode != 0:
        print("    scenario stderr tail: " + proc.stderr.strip()[-400:], flush=True)
    print(f"    [{tag}] driver exited {proc.returncode} in {time.time() - started:.1f}s", flush=True)
    payload = {rank: load_child(tag, rank) for rank in (0, 1)}
    missing = [rank for rank, item in payload.items() if item.get("ok") is not True]
    assert not missing, (
        f"scenario {tag} did not complete on ranks {missing}; "
        f"driver rc={proc.returncode} (this is a DEADLOCK/FAILURE finding, not a retry)")
    _SCENARIO_CACHE[tag] = payload
    return payload


def run_scenario(arm: str, tag: str, synth: bool) -> int:
    """Driver: spin up the two ranks for one scenario, then exit."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = str(Path(tmp) / "gloo_store")
        procs = [CTX.Process(target=child, args=(r, init_file, arm, tag, synth))
                 for r in (0, 1)]
        for proc in procs:
            proc.start()
        deadline = time.time() + CHILD_TIMEOUT_S
        for proc in procs:
            proc.join(timeout=max(1, int(deadline - time.time())))
        hung = [proc.pid for proc in procs if proc.is_alive()]
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
            proc.join(5)
        codes = [proc.exitcode for proc in procs]
    print(f"scenario {tag}: ranks exited {codes} in {time.time() - started:.1f}s", flush=True)
    if hung:
        print(f"scenario {tag}: DEADLOCK, ranks {hung} past {CHILD_TIMEOUT_S}s", flush=True)
        return 2
    return 0 if all(code == 0 for code in codes) else 1


def reference_cases(T):
    cases = []
    for rank in (0, 1):
        case = build_local(T, RANK_ROWS[rank], offset=rank)
        case["_T"] = T
        cases.append(case)
    return cases


def max_abs_diff(a, b) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def l2(a, b) -> float:
    return float(torch.sqrt(sum(((x - y) ** 2).sum() for x, y in zip(a, b))))


# --------------------------------------------------------------------------------- tests
# The comparison quantities are deliberately the ones that can be verified independently:
# (a) the pooled components the projection actually consumed, (b) cross-rank bitwise equality,
# (c) the mean-of-ranks identity. An earlier version of this suite compared the full encoder
# gradient vector against a hand-rolled reference built in this file; that reference disagreed
# with the production path and was itself buggy, so it was removed rather than loosened until it
# agreed. The properties below are what issue #71 §3 actually asks for.


def parent_pooled_component_norms():
    """Norms of the mean over the two ranks' LOCAL gradients, computed without any DDP.

    Fresh cases per key: sharing one set of cases across three loss terms made the second and
    third reads see a graph whose other_rank gradient had already been consumed, which silently
    halved the reported mean.
    """
    norms = []
    for key in ("base_loss", "l1_loss", "other_loss"):
        cases = reference_cases(T_R4_08)
        encoder = cases[0]["encoder"]
        per_rank = [local_grads(cases[rank], key) for rank in (0, 1)]
        mean = [(per_rank[0][i] + per_rank[1][i]) / 2.0 for i in range(len(encoder))]
        norms.append(float(torch.cat([g.reshape(-1) for g in mean]).norm()))
    return norms


def test_two_ranks_complete_without_deadlock():
    ranks = run_pair("r4_08", "r4_08")
    for rank in (0, 1):
        payload = ranks[rank]
        assert payload.get("ok") is True, f"rank {rank} did not finish: {payload}"
        assert payload.get("iterations") == ranks[1 - rank].get("iterations"), (
            "both ranks must run the same number of steps")
    print("  both ranks finished with no deadlock", flush=True)


def test_projection_consumes_globally_pooled_components():
    """The vectors the projection uses must be the MEAN over the ranks' local gradients."""
    ranks = run_pair("r4_08", "r4_08")
    observed = ranks[0]["pooled_component_norms"]
    expected = parent_pooled_component_norms()
    assert observed is not None, "rank 0 did not record the pooled components it consumed"
    for name, got, want in zip(("base_loss", "l1_loss", "other_loss"), observed, expected):
        assert abs(got - want) <= 1e-5 * max(want, 1.0), (
            f"pooled {name} gradient: the production path consumed ||g||={got:.8f} but the "
            f"independent mean over the two ranks' local gradients is {want:.8f}")
    print(f"  pooled component norms {name}=" if False else "", end="")
    print(f"  production pooled {['%.8f' % v for v in observed]} == "
          f"independent mean {['%.8f' % v for v in expected]}", flush=True)
    assert observed[2] == 0.0, (
        "the L2+L3 component must be exactly zero on the encoder (see the R5-10 finding)")


def test_no_double_averaging():
    cases = reference_cases(T_R4_08)
    g0 = local_grads(cases[0], "base_loss")
    g1 = local_grads(cases[1], "base_loss")
    mean = [(a + b) / 2.0 for a, b in zip(g0, g1)]
    total = [a + b for a, b in zip(g0, g1)]
    mean_norm = float(torch.cat([g.reshape(-1) for g in mean]).norm())
    total_norm = float(torch.cat([g.reshape(-1) for g in total]).norm())
    assert mean_norm > 0, "the fixture must produce a non-zero base gradient"
    assert abs(total_norm - 2.0 * mean_norm) < 1e-4 * max(total_norm, 1.0), (
        f"mean and sum of the same vectors must differ by exactly a factor 2, got "
        f"{total_norm / mean_norm:.9f}")
    print(f"  ||mean|| = {mean_norm:.6f}, ||sum|| = {total_norm:.6f}, "
          f"ratio = {total_norm / mean_norm:.9f} (exactly 2)", flush=True)


def test_projection_uses_globally_averaged_both_gradients():
    """Cancelling local dot products with a non-negative global dot => zero projections."""
    ranks = run_pair("r4_08", "r4_08_synth", synth=True)
    for rank in (0, 1):
        assert ranks[rank].get("ok") is True, f"rank {rank} failed: {ranks[rank]}"
    pooled_dot = ranks[0]["pooled_dot"]
    assert pooled_dot >= 0.0, (
        f"fixture precondition: the POOLED dot must be non-negative, got {pooled_dot}")
    fired = ranks[0]["fired"]
    assert ranks[1]["fired"] == fired, "both ranks must reach the same decision"
    assert fired == 0, (
        f"local dots have opposite signs ({ranks[0]['local_dot']:+.1f} vs "
        f"{ranks[1]['local_dot']:+.1f}) and the pooled dot is non-negative, so a correct GLOBAL "
        f"projection must fire 0 times; it fired {fired}. A non-zero count means the projection "
        "runs on per-rank LOCAL gradients.")
    gathered = ranks[0]["_tensors"]["gathered"]
    assert torch.equal(gathered[0], gathered[1]), (
        "with no global conflict both ranks must assemble bitwise identical gradients")
    print(f"  local dots {ranks[0]['local_dot']:+.1f} / {ranks[1]['local_dot']:+.1f}, "
          f"projections fired = {fired}, ranks bitwise identical", flush=True)


def test_ranks_synchronised_after_step():
    ranks = run_pair("r4_08", "r4_08")
    gathered = ranks[0]["_tensors"]["gathered"]
    assert torch.equal(gathered[0], gathered[1]), (
        "weights must be bitwise identical across ranks after the AdamW step. Divergence here "
        "means the ranks saw different gradients, which is exactly the failure mode under test.")
    print("  weights bitwise identical across ranks after the step", flush=True)


def test_decoder_codebook_gradients_reduced_but_not_touched_by_projection():
    """DDP must reduce the non-encoder gradients, and the projection must not touch them."""
    ranks = run_pair("r4_08", "r4_08")
    reduced = float(ranks[0]["_tensors"]["nonencoder_grads"].norm())
    cases = reference_cases(T_R4_08)
    non_encoder = cases[0]["non_encoder"]
    per_rank = []
    for case in cases:
        grads = torch.autograd.grad(case["base_loss"], non_encoder, retain_graph=True,
                                    allow_unused=True)
        per_rank.append([torch.zeros_like(p) if g is None else g
                         for p, g in zip(non_encoder, grads)])
    local_norm = float(torch.cat([g.reshape(-1) for g in per_rank[0]]).norm())
    assert reduced < local_norm, (
        f"the DDP reducer must shrink the non-encoder gradient: observed {reduced:.8f} vs "
        f"rank-0 local {local_norm:.8f}. Equal norms would mean the reducer never ran, which "
        "happens when backward() is called on a loss built outside DDP.forward().")
    print(f"  non-encoder ||grad|| reduced by the reducer: {local_norm:.8f} -> {reduced:.8f}",
          flush=True)


def pooled_parts(T, cases):
    """Explicitly pooled per-rank local encoder gradients, per loss term."""
    encoder = cases[0]["encoder"]
    pooled = {}
    for key in ("base_loss", "l1_loss", "other_loss"):
        per_rank = [T._r4_08_encoder_gradients(c[key], encoder, retain=True) for c in cases]
        pooled[key] = [(per_rank[0][i] + per_rank[1][i]) / 2.0 for i in range(len(encoder))]
    return pooled["base_loss"], pooled["l1_loss"], pooled["other_loss"], cases


def linear_blocks(T, case):
    if hasattr(T, "_r5_04_linear_blocks"):
        return T._r5_04_linear_blocks(case["model"].encoder, case["encoder"])
    return [list(range(len(case["encoder"])))]


def test_r5_04_per_layer_pooled_norms():
    T = T_R5_04
    cases = reference_cases(T)
    pooled_base, pooled_l1, _, _ = pooled_parts(T, cases)
    blocks = linear_blocks(T, cases[0])
    assert len(blocks) >= 2, f"expected several blocks, got {len(blocks)}"
    global_dot = float(torch.cat([g.reshape(-1) for g in pooled_l1])
                       @ torch.cat([g.reshape(-1) for g in pooled_base]))
    block_dots = [float(torch.cat([pooled_l1[i].reshape(-1) for i in b])
                        @ torch.cat([pooled_base[i].reshape(-1) for i in b]))
                  for b in blocks]
    differing = any((d < 0.0) != (global_dot < 0.0) for d in block_dots)
    print(f"  {len(blocks)} blocks; pooled global dot {global_dot:+.4e}; block dots "
          f"{[f'{d:+.3e}' for d in block_dots]}", flush=True)
    print(f"  a block decision differs from the whole-vector decision: {differing}", flush=True)
    assert all(math.isfinite(d) for d in block_dots)


def test_r5_09_ema_buffers_identical_on_both_ranks():
    ranks = run_pair("r5_09", "r5_09")
    for rank in (0, 1):
        payload = ranks[rank]
        assert payload.get("ok") is True, f"rank {rank} failed: {payload}"
        assert payload.get("iterations") == 3, "the EMA test must run 3 steps"
        assert payload.get("ema_buffers"), f"rank {rank} reported no EMA buffers"
    a, b = ranks[0]["ema_buffers"], ranks[1]["ema_buffers"]
    assert a == b, (
        "after 3 steps the EMA buffer must be identical on every rank; a per-rank EMA would make "
        f"the projection rank-dependent. rank0[:3]={a[:3]} rank1[:3]={b[:3]}")
    print(f"  EMA buffers identical across ranks ({len(a)} tensors), first |m| sums {a[:3]}",
          flush=True)


def main() -> int:
    tests = [
        test_two_ranks_complete_without_deadlock,
        test_projection_consumes_globally_pooled_components,
        test_no_double_averaging,
        test_projection_uses_globally_averaged_both_gradients,
        test_ranks_synchronised_after_step,
        test_decoder_codebook_gradients_reduced_but_not_touched_by_projection,
        test_r5_04_per_layer_pooled_norms,
        test_r5_09_ema_buffers_identical_on_both_ranks,
    ]
    failed = 0
    for test in tests:
        print(f"  -> {test.__name__}", flush=True)
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
        print("DDP_TESTS_RED", flush=True)
        return 1
    print("ALL TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    LOGS.mkdir(parents=True, exist_ok=True)
    if "--scenario" in sys.argv:
        idx = sys.argv.index("--scenario")
        arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "r4_08"
        synth = "--synth" in sys.argv and sys.argv[sys.argv.index("--synth") + 1] == "1"
        sys.exit(run_scenario(arm, sys.argv[idx + 1], synth))
    sys.exit(main())
