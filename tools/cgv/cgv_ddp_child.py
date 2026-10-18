"""G5 driver: a real two-rank gloo DDP run compared against an independent pooled reference.

Runs as its own process (``cgv.engine`` spawns it) so the gloo store is created and torn down in
a pristine interpreter. Each rank persists its result to ``payload/`` because a forked child gets
a COPY of the parent's memory, so nothing it writes to a module global reaches the parent.

Usage: cgv_ddp_child.py <tree> <payload_dir> <timeout_s>
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

SEED = 2024
IN_DIM = 32
N_SRC = 24
BEHAVIOR_WEIGHT = 0.5
RANK_ROWS = {0: 24, 1: 20}
CTX = mp.get_context("fork")


def _arm_kwargs(T, case):
    import inspect

    accepted = inspect.signature(T._r4_08_apply_projection).parameters
    kwargs = {}
    if "recon_loss" in accepted:
        kwargs["recon_loss"] = case["recon_loss"]
    if "encoder_module" in accepted:
        kwargs["encoder_module"] = case["model"].encoder
    return kwargs


def _make_batch(rows: int, offset: int):
    generator = torch.Generator().manual_seed(SEED + 100 * offset + rows)
    emb = torch.randn(2 * rows, IN_DIM, generator=generator)
    source_ids = torch.arange(rows)
    target_ids = (torch.arange(rows) + 1) % rows
    return emb, source_ids, target_ids


def _build_local(T, rows: int, offset: int):
    torch.manual_seed(SEED)
    model = T.RQVAE(T._tokenizer_config(), in_dim=IN_DIM)
    emb, source_ids, target_ids = _make_batch(rows, offset)
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


def _local_grads(T, case, key: str):
    return T._r4_08_encoder_gradients(case[key], case["encoder"], retain=True)


def child(rank: int, init_file: str, module_name: str, tree: str, payload: str) -> None:
    result: dict = {"rank": rank, "ok": False}
    tensors: dict[str, torch.Tensor] = {}
    try:
        sys.path.insert(0, tree)
        import importlib.util

        spec = importlib.util.spec_from_file_location(module_name, Path(tree) / "train_rqvae.py")
        T = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = T
        try:
            spec.loader.exec_module(T)
        except RuntimeError:
            import ast

            entry = Path(tree) / "train_rqvae.py"
            parsed = ast.parse(entry.read_text(encoding="utf-8"), filename=str(entry))
            parsed.body = [
                n for n in parsed.body
                if not (isinstance(n, ast.If)
                        and ast.unparse(n.test).startswith("EXPERIMENT_ARM !="))]
            T = type(sys)(module_name)
            T.__file__ = str(entry)
            sys.modules[module_name] = T
            exec(compile(parsed, str(entry), "exec"), T.__dict__)  # noqa: S102

        dist.init_process_group(backend="gloo", init_method=f"file://{init_file}",
                                rank=rank, world_size=2)
        rows = RANK_ROWS[rank]
        base_case = _build_local(T, rows, offset=rank)
        model = base_case["model"]
        wrapped = DDP(model, device_ids=None, find_unused_parameters=False)
        case = dict(base_case)
        emb, source_ids, target_ids = _make_batch(rows, rank)
        reconstructed, quant_loss, _, _ = wrapped(emb)
        case["base_loss"], case["recon_loss"] = model.compute_loss(emb, reconstructed, quant_loss)
        residuals = T._residuals_with_trace(model, emb)
        (case["l1_loss"], case["other_loss"],
         case["valid_levels"]) = T.behavior_contrastive_components(
            residuals, source_ids, target_ids, None)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        model.zero_grad(set_to_none=False)
        case["base_loss"].backward(retain_graph=True)
        pooled = []
        for key in ("base_loss", "l1_loss", "other_loss"):
            grads = T._r4_08_global_average(_local_grads(T, case, key))
            pooled.append(float(torch.cat([g.reshape(-1) for g in grads]).norm()))
        T._r4_08_apply_projection(case["base_loss"], case["l1_loss"], case["other_loss"],
                                  case["encoder"], BEHAVIOR_WEIGHT,
                                  valid_levels=case["valid_levels"], **_arm_kwargs(T, case))
        if rank == 0:
            tensors["encoder_grads"] = torch.cat(
                [p.grad.detach().reshape(-1).clone() for p in case["encoder"]])
            tensors["nonencoder_grads"] = torch.cat(
                [p.grad.detach().reshape(-1).clone() for p in case["non_encoder"]])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        flat = torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])
        gathered = [torch.zeros_like(flat) for _ in range(2)]
        dist.all_gather(gathered, flat)
        tensors["gathered"] = torch.stack(gathered)
        result = {"rank": rank, "ok": True, "rows": rows, "pooled_component_norms": pooled}
    except Exception as exc:  # noqa: BLE001
        result = {"rank": rank, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        out = Path(payload)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"rank{rank}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if tensors:
            torch.save(tensors, out / f"rank{rank}.pt")


def _independent_reference(T, tree_payload: Path):
    """Mean over the two ranks' LOCAL gradients, computed in one process without DDP."""
    cases = [_build_local(T, RANK_ROWS[r], offset=r) for r in (0, 1)]
    encoder = cases[0]["encoder"]
    pooled = {}
    for key in ("base_loss", "l1_loss", "other_loss"):
        per_rank = [_local_grads(T, c, key) for c in cases]
        pooled[key] = [(per_rank[0][i] + per_rank[1][i]) / 2.0 for i in range(len(encoder))]
    return [float(torch.cat([g.reshape(-1) for g in pooled[k]]).norm())
            for k in ("base_loss", "l1_loss", "other_loss")]


def main(tree: str, payload: str, timeout_s: int) -> int:
    payload_dir = Path(payload)
    payload_dir.mkdir(parents=True, exist_ok=True)
    module_name = f"cgv_g5_{Path(tree).name.replace('-', '_')}"
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        init_file = str(Path(tmp) / "gloo_store")
        procs = [CTX.Process(target=child, args=(r, init_file, module_name, tree, payload))
                 for r in (0, 1)]
        for p in procs:
            p.start()
        deadline = time.time() + timeout_s
        for p in procs:
            p.join(timeout=max(1, int(deadline - time.time())))
        hung = [p.pid for p in procs if p.is_alive()]
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(5)
    elapsed = round(time.time() - started, 1)
    ranks = []
    for r in (0, 1):
        path = payload_dir / f"rank{r}.json"
        ranks.append(json.loads(path.read_text()) if path.is_file()
                     else {"rank": r, "ok": False, "error": "no result file"})
    if hung or any(not r.get("ok") for r in ranks):
        summary = {
            "passed": False,
            "detail": f"the two-rank run did not complete (hung={hung}, ranks={ranks})",
            "measurements": {"ranks": ranks, "elapsed_s": elapsed},
            "finding": "G5 could not execute the two-rank path; treated as a failure, not a skip",
        }
        (payload_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)
        return 1

    tensors0 = torch.load(payload_dir / "rank0.pt", weights_only=False)
    sys.path.insert(0, tree)
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, Path(tree) / "train_rqvae.py")
    T = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = T
    try:
        spec.loader.exec_module(T)
    except RuntimeError:
        import ast

        entry = Path(tree) / "train_rqvae.py"
        parsed = ast.parse(entry.read_text(encoding="utf-8"), filename=str(entry))
        parsed.body = [
            n for n in parsed.body
            if not (isinstance(n, ast.If)
                    and ast.unparse(n.test).startswith("EXPERIMENT_ARM !="))]
        T = type(sys)(module_name)
        T.__file__ = str(entry)
        sys.modules[module_name] = T
        exec(compile(parsed, str(entry), "exec"), T.__dict__)  # noqa: S102
    reference = _independent_reference(T, payload_dir)

    pooled_match = all(abs(a - b) <= 1e-5 * max(abs(b), 1.0)
                       for a, b in zip(ranks[0]["pooled_component_norms"], reference))
    gathered = tensors0["gathered"]
    ranks_identical = bool(torch.equal(gathered[0], gathered[1]))
    passed = pooled_match and ranks_identical
    summary = {
        "passed": passed,
        "detail": (f"the projection consumed the mean over the ranks' local gradients "
                   f"(production {['%.8f' % v for v in ranks[0]['pooled_component_norms']]} == "
                   f"independent {['%.8f' % v for v in reference]}) and both ranks hold "
                   f"bitwise identical weights after the AdamW step"
                   if passed else
                   (f"pooled components match={pooled_match}, ranks identical="
                    f"{ranks_identical}")),
        "measurements": {
            "production_pooled_component_norms": ranks[0]["pooled_component_norms"],
            "independent_pooled_component_norms": reference,
            "pooled_components_match": pooled_match,
            "ranks_bitwise_identical": ranks_identical,
            "local_batch_rows": {str(r): RANK_ROWS[r] for r in (0, 1)},
            "elapsed_s": elapsed,
        },
        "finding": None if passed else
        ("gradients are not globally pooled (double averaging or per-rank PCGrad) or the ranks "
         "diverged"),
    }
    (payload_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
