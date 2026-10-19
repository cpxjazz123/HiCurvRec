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


def _rank(rank: int, tree: str, payload: str, world_size: int, spec_text: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    out = Path(payload)
    try:
        contract = C.normalise_spec(spec_text, source="the declaration on the child's stdin")
        trainer, saved_path, saved_mods = E.load_trainer(Path(tree), module_name="cgv_rank")
        try:
            seed = 1000 + rank
            # DDP broadcasts parameters at construction, so the wrapper is created FIRST and every
            # per-rank gradient is then taken at that same synchronised parameter point. Taking
            # the local gradients before the broadcast would compare gradients evaluated at
            # different parameters, and the difference would look like an aggregation failure.
            base = adapters.build_observations(trainer, contract, seed=1000)
            model = base["model"]
            ddp = DDP(model, device_ids=None)

            # the rank's own view: plain forward, term gradients computed WITHOUT DDP averaging
            local_obs = adapters.build_observations(trainer, contract, seed=seed, model=model)
            terms = adapters.loss_terms(trainer, contract, local_obs)
            local = {name: [tensor.detach().clone()
                            for tensor in adapters.term_gradient(trainer, local_obs, terms[name])]
                     for name in terms}
            torch.save(local, out / f"rank_{rank}_local.pt")

            # the production path: a real DDP wrapper, so the framework performs the declared
            # aggregation during backward, and the mechanism then runs on the averaged gradient
            observations = adapters.build_observations(trainer, contract, seed=seed, model=model)
            reconstructed, quant_loss, _, _ = ddp(observations["embeddings"])
            base_loss, recon_loss = model.compute_loss(observations["embeddings"], reconstructed,
                                                       quant_loss)
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
            production = adapters.executed_target_gradient(trainer, contract, observations)
            torch.save(production, out / f"rank_{rank}_production.pt")

            # one declared optimizer step, so the ranks can be compared afterwards
            update = contract["update_contract"]
            optimizer_cls = getattr(torch.optim, update["optimizer"])
            optimizer = optimizer_cls(model.parameters(), lr=float(update.get("lr", 1e-3)),
                                      weight_decay=float(update.get("weight_decay", 0.01)))
            optimizer.step()
            digest = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).sum().item()
            rows = int(observations["embeddings"].shape[0])

            gathered: list[Any] = [None] * world_size
            dist.all_gather_object(gathered, {"digest": digest, "rows": rows})
            if rank == 0:
                gradient = contract["gradient_contract"]
                oracle_inputs = {
                    "scale": float(observations["scale"]),
                    "rule": gradient["rule"],
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
                pooled = [torch.zeros_like(tensor) for tensor in production]
                for source in range(world_size):
                    piece = torch.load(out / f"rank_{source}_production.pt", weights_only=False)
                    for index, tensor in enumerate(piece):
                        pooled[index] = pooled[index] + tensor / world_size
                torch.save(pooled, out / "production_pooled.pt")
                digests = [item["digest"] for item in gathered]
                (out / "summary.json").write_text(json.dumps({
                    "ok": True,
                    "weights": None,
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
    spec_text = sys.stdin.read()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + (os.getpid() % 500)))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    summary = Path(payload) / "summary.json"
    if summary.is_file():
        summary.unlink()
    try:
        mp.spawn(_rank, args=(tree, payload, int(world_size), spec_text), nprocs=int(world_size),
                 join=True)
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
