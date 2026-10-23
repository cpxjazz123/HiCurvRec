"""Real multi-rank execution for CGV's distributed gate (G5).

Runs the declared number of ranks with gloo, executes the mechanism's real assembly on each rank
through a genuine ``DistributedDataParallel`` wrapper, and hands the parent everything it needs to
judge the result INDEPENDENTLY:

    rank_<i>_local.pt     the rank's own term gradients, computed without DDP averaging
    production_pooled.pt  the gradient the real DDP + mechanism path produced
    rank_<i>.json         rank metadata (batch rows, parameter digest)

The parent pools the local gradients with the independent aggregation oracle and applies the
independent gradient oracle to the pooled result, then compares that with ``production_pooled``.
The child deliberately does NOT decide whether the aggregation is correct: if it did, a wrong
aggregation would be checked against itself.

The declaration is passed on stdin, for the same reason the engine takes it as an argument: the
tree supplies no declaration of its own.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters  # noqa: E402
import contract as C  # noqa: E402
import engine as E  # noqa: E402


def _install_injected_fault() -> str | None:
    """A fault that concerns the COLLECTIVE has to be installed here, in the process that runs it.

    The parent cannot patch this process's collectives, so the suite declares the fault by
    environment variable and this child applies it. Silently skipping it would let a broken
    aggregation pass a gate that exists precisely to catch one.
    """
    import os as _os

    fault = _os.environ.get("CGV_INJECTED_FAULT")
    if fault != "wrong_ddp_reduce":
        return None
    original_reduce = dist.reduce

    def wrong_reduce(tensor, dst=None, op=None, group=None, async_op=False):
        # summing instead of averaging: exactly the double-counting bug the gate looks for
        return original_reduce(tensor, dst=dst, op=dist.ReduceOp.SUM, group=group,
                               async_op=async_op)

    dist.reduce = wrong_reduce
    original_all_reduce = dist.all_reduce

    def wrong_all_reduce(tensor, op=None, group=None, async_op=False):
        return original_reduce(tensor, dst=None, op=dist.ReduceOp.SUM, group=group,
                               async_op=async_op)

    dist.all_reduce = wrong_all_reduce
    return fault


def _rank(rank: int, tree: str, payload: str, world_size: int, spec_text: str,
          rank_sizes=None) -> None:
    torch.set_num_threads(1)
    _install_injected_fault()
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    out = Path(payload)
    trace_reduce_calls = 0
    try:
        contract = C.normalise_spec(spec_text, source="the declaration on the child's stdin")
        trainer, saved_path, saved_mods = E.load_trainer(Path(tree), module_name="cgv_rank")
        try:
            seed = 1000 + rank
            # DDP broadcasts parameters at construction, so the wrapper is created FIRST and every
            # per-rank gradient is then taken at that same synchronised parameter point. Taking
            # the local gradients before the broadcast would compare gradients evaluated at
            # different parameters, and the difference would look like an aggregation failure.
            trace_mode = E.is_trace_mechanism(contract)
            model = adapters.build_model(trainer, contract, seed=1000)
            ddp = DDP(model, device_ids=None)

            # the rank's own view: term gradients computed WITHOUT DDP averaging, so the parent
            # can pool them independently and compare that against what DDP actually produced
            local_n = (rank_sizes[rank] if rank_sizes and rank < len(rank_sizes) else None)
            if trace_mode:
                local_obs = adapters.trace_case(trainer, contract, seed=seed, model=model,
                                                n_src=local_n)
                local_n_src = local_obs["n_src"]
                src = torch.arange(local_n_src)
                tgt = (torch.arange(local_n_src) + 1) % local_n_src
                reconstructed_l, quant_l, _, _ = model.forward(local_obs["embeddings"])
                base_l, _ = model.compute_loss(local_obs["embeddings"], reconstructed_l, quant_l)
                stacked_l = trainer._residuals_with_trace(model, local_obs["embeddings"])
                losses_l = []
                for trace_tensor in stacked_l:
                    logits = -torch.cdist(trace_tensor[:local_n_src], trace_tensor[local_n_src:],
                                          p=2) / local_obs["temperature"]
                    losses_l.append(trainer._contrastive_ce(logits, src, tgt, None))
                local_total = base_l + float(local_obs["weight"]) * torch.stack(losses_l).mean()
                local_params = local_obs["encoder"]
                local_grads = torch.autograd.grad(local_total, local_params, retain_graph=True,
                                                  allow_unused=True)
                local = {"encoder": [torch.zeros_like(p) if g is None else g.detach().clone()
                                     for p, g in zip(local_params, local_grads)]}
                if local_obs.get("codebooks_parameters"):
                    codebook_params = local_obs["codebooks_parameters"]
                    codebook_grads = torch.autograd.grad(local_total, codebook_params,
                                                         retain_graph=False, allow_unused=True)
                    local["codebooks"] = [torch.zeros_like(p) if g is None else g.detach().clone()
                                          for p, g in zip(codebook_params, codebook_grads)]
                torch.save(local, out / f"rank_{rank}_local.pt")
            else:
                local_obs = adapters.build_observations(trainer, contract, seed=seed, model=model,
                                                        n_src=local_n)
                terms = adapters.loss_terms(trainer, contract, local_obs)
                local = {name: [tensor.detach().clone()
                                for tensor in adapters.term_gradient(trainer, local_obs,
                                                                    terms[name])]
                         for name in terms}
                torch.save(local, out / f"rank_{rank}_local.pt")
                # The codebook's own per-rank gradients. A mechanism that targets the codebook
                # moves a gradient the encoder route cannot see, so the parent needs the
                # codebook's local gradients AND the mechanism's own contribution to compare
                # its independent pooling against what the distributed path produced.
                codebook_params = adapters.optional_parameter_group(
                    trainer, local_obs, contract, "codebook")
                if codebook_params is not None:
                    local_codebook = {
                        name: adapters.term_gradient_for(trainer, local_obs, terms[name],
                                                          codebook_params)
                        for name in terms
                    }
                    local_codebook["mechanism"] = adapters.mechanism_parameter_contribution(
                        trainer, contract, local_obs, codebook_params)
                    torch.save(local_codebook, out / f"rank_{rank}_codebook_local.pt")

            # the production path: a real DDP wrapper, so the framework performs the declared
            # aggregation during backward, and the mechanism then runs on the averaged gradient
            if trace_mode:
                observations = adapters.trace_case(trainer, contract, seed=seed, model=model,
                                                   n_src=local_n)
                n_src = observations["n_src"]
                source_ids = torch.arange(n_src)
                target_ids = (torch.arange(n_src) + 1) % n_src

                # Build the production graph the way the training loop does: the reconstruction
                # loss through the DDP wrapper, the behaviour loss from the raw module, and ONE
                # backward over their sum. Arming DDP with a throwaway forward instead would put
                # the reducer's sink outside the loss graph, and the behaviour contribution would
                # silently escape the reduction -- a different training regime, not a stricter one.
                reduce_calls = 0
                original_all_reduce = dist.all_reduce

                def counting_all_reduce(*args, **kwargs):
                    nonlocal reduce_calls
                    reduce_calls += 1
                    return original_all_reduce(*args, **kwargs)

                dist.all_reduce = counting_all_reduce
                try:
                    reconstructed, quant_loss, _, _ = ddp(observations["embeddings"])
                    base_loss, recon_loss = model.compute_loss(observations["embeddings"],
                                                               reconstructed, quant_loss)
                    stacked = trainer._residuals_with_trace(model, observations["embeddings"])
                    level_losses = []
                    for trace_tensor in stacked:
                        logits = -torch.cdist(trace_tensor[:n_src], trace_tensor[n_src:], p=2) \
                            / observations["temperature"]
                        level_losses.append(trainer._contrastive_ce(logits, source_ids, target_ids,
                                                                    None))
                    behaviour = torch.stack(level_losses).mean()
                    weight = float(observations["weight"])
                    total = base_loss + weight * behaviour
                    model.zero_grad(set_to_none=False)
                    total.backward()
                finally:
                    dist.all_reduce = original_all_reduce
                    trace_reduce_calls = reduce_calls

                encoder = observations["encoder"]
                production = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                              for p in encoder]
                payload_out = {"encoder": production}
                if observations.get("codebooks_parameters"):
                    payload_out["codebooks"] = [
                        torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                        for p in observations["codebooks_parameters"]]
                torch.save(payload_out, out / f"rank_{rank}_production.pt")
                if rank == 0:
                    (out / "reduce_calls.json").write_text(json.dumps(
                        {"reduce_calls": reduce_calls}))
            else:
                observations = adapters.build_observations(trainer, contract, seed=seed, model=model,
                                                           n_src=local_n)
                reconstructed, quant_loss, _, _ = ddp(observations["embeddings"])
                base_loss, recon_loss = model.compute_loss(observations["embeddings"],
                                                           reconstructed, quant_loss)
                residuals = trainer._residuals_with_trace(model, observations["embeddings"])
                n_src = observations["n_src"]
                l1_loss, other_loss, valid = trainer.behavior_contrastive_components(
                    residuals, torch.arange(n_src), (torch.arange(n_src) + 1) % n_src, None)
                observations.update({
                    "ddp": ddp,
                    "base_loss": base_loss,
                    "recon_loss": recon_loss,
                    "quant_loss": quant_loss,
                    "l1_loss": l1_loss,
                    "other_loss": other_loss,
                    "valid_levels": int(valid),
                    "scale": observations["weight"] / max(int(valid), 1),
                })
                # count the reducer's collectives on the NON-trace path too: reporting zero
                # here would read as "DDP never reduced", which is a different claim from
                # "this path does not count". A double reduction would show up as two.
                reduce_calls = 0
                original_all_reduce = dist.all_reduce

                def counting_all_reduce(*args, **kwargs):
                    nonlocal reduce_calls
                    reduce_calls += 1
                    return original_all_reduce(*args, **kwargs)

                dist.all_reduce = counting_all_reduce
                try:
                    production = adapters.executed_target_gradient(trainer, contract, observations)
                finally:
                    dist.all_reduce = original_all_reduce
                    trace_reduce_calls = reduce_calls
                torch.save(production, out / f"rank_{rank}_production.pt")
                codebook_params = adapters.optional_parameter_group(
                    trainer, observations, contract, "codebook")
                if codebook_params is not None:
                    production_codebook = [
                        torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                        for p in codebook_params
                    ]
                    torch.save(production_codebook,
                               out / f"rank_{rank}_production_codebook.pt")

            # one declared optimizer step, so the ranks can be compared afterwards
            update = contract["update_contract"]
            optimizer_cls = getattr(torch.optim, update["optimizer"])
            optimizer = optimizer_cls(model.parameters(), lr=float(update.get("lr", 1e-3)),
                                      weight_decay=float(update.get("weight_decay", 0.01)))
            optimizer.step()
            digest = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).sum().item()
            rows = int(observations["embeddings"].shape[0])

            # a double reduction would show up here as more than one collective per step
            reduce_calls = trace_reduce_calls if rank == 0 else 0
            gathered: list[Any] = [None] * world_size
            dist.all_gather_object(gathered, {"digest": digest, "rows": rows})
            if rank == 0:
                gradient = contract["gradient_contract"]
                oracle_inputs = {
                    "scale": float(observations["scale"]),
                    "rule": gradient.get("rule"),
                    "reference_term": gradient.get("reference_term"),
                    "alpha": float(adapters.declared_value(trainer, gradient, "alpha", 1.0)),
                    "eps": float(adapters.declared_value(trainer, gradient, "eps", 1e-12)),
                    "gate_open": True,
                }
                if gradient.get("cos_threshold_flag"):
                    oracle_inputs["cos_threshold"] = float(
                        adapters.declared_value(trainer, gradient, "cos_threshold", -0.2))
                if gradient.get("blocks"):
                    oracle_inputs["blocks"] = adapters.structural_blocks(
                        trainer, contract, observations)
                (out / "oracle_inputs.json").write_text(json.dumps(oracle_inputs, indent=2))
                # mean over the ranks, which is what DDP already applied per parameter; this is
                # the value the parent then checks against its OWN pooling of the local gradients
                first_piece = torch.load(out / "rank_0_production.pt", weights_only=False)
                if isinstance(first_piece, dict):
                    pooled = {
                        key: [torch.zeros_like(tensor) for tensor in value]
                        for key, value in first_piece.items()
                    }
                    for source in range(world_size):
                        piece = torch.load(out / f"rank_{source}_production.pt", weights_only=False)
                        for key in pooled:
                            for index, tensor in enumerate(piece[key]):
                                pooled[key][index] = pooled[key][index] + tensor / world_size
                else:
                    pooled = [torch.zeros_like(tensor) for tensor in first_piece]
                    for source in range(world_size):
                        piece = torch.load(out / f"rank_{source}_production.pt", weights_only=False)
                        for index, tensor in enumerate(piece):
                            pooled[index] = pooled[index] + tensor / world_size
                torch.save(pooled, out / "production_pooled.pt")
                # the same pooling for the codebook's production gradient, so the parent has a
                # DDP-averaged codebook gradient to compare its OWN pooling against
                codebook_first = out / "rank_0_production_codebook.pt"
                if codebook_first.is_file():
                    template = torch.load(codebook_first, weights_only=False)
                    pooled_codebook = [torch.zeros_like(tensor) for tensor in template]
                    for source in range(world_size):
                        piece = torch.load(out / f"rank_{source}_production_codebook.pt",
                                           weights_only=False)
                        for index, tensor in enumerate(piece):
                            pooled_codebook[index] = pooled_codebook[index] + tensor / world_size
                    torch.save(pooled_codebook, out / "production_codebook_pooled.pt")
                digests = [item["digest"] for item in gathered]
                (out / "summary.json").write_text(json.dumps({
                    "ok": True,
                    "weights": None,
                    "reduce_calls": reduce_calls,
                    "ranks_identical": len(set(digests)) == 1,
                    "local_batch_rows": {str(i): gathered[i]["rows"] for i in range(world_size)},
                    "elapsed_s": round(time.time() - started_at, 1),
                }, indent=2))
        finally:
            E.unload_trainer(saved_path, saved_mods)
    except Exception:  # noqa: BLE001
        (out / f"rank_{rank}_error.txt").write_text(traceback.format_exc())
        if rank == 0:
            (out / "summary.json").write_text(json.dumps({
                "ok": False,
                "error": traceback.format_exc().strip().splitlines()[-1],
                "measurements": {"traceback": traceback.format_exc()[-2000:]},
            }, indent=2))
        raise
    finally:
        dist.destroy_process_group()


started_at = time.time()


def main(tree: str, payload: str, world_size: int = 2, timeout: int = 900) -> int:
    import json as _json

    envelope = _json.loads(sys.stdin.read() or "{}")
    spec_text = _json.dumps(envelope["spec"])
    rank_sizes = envelope.get("rank_batch_sizes")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + (os.getpid() % 500)))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    summary = Path(payload) / "summary.json"
    if summary.is_file():
        summary.unlink()
    try:
        mp.spawn(_rank, args=(tree, payload, int(world_size), spec_text, rank_sizes),
                 nprocs=int(world_size), join=True)
    except Exception:  # noqa: BLE001
        if not summary.is_file():
            summary.write_text(json.dumps({
                "ok": False, "error": "the rank processes failed",
                "measurements": {"traceback": traceback.format_exc()[-2000:]},
            }, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2],
                  int(sys.argv[3]) if len(sys.argv) > 3 else 2,
                  int(sys.argv[4]) if len(sys.argv) > 4 else 900))
