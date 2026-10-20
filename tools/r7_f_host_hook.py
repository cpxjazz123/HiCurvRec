"""Host-native, OFF-by-default auxiliary loss callback for the real Stage2 trainer (issue #88).

The #87 pilot was invalid because it replayed frozen bundles instead of running the production
loop: no real behaviour-CE minibatch, no real false-negative mask, and a soft posterior measured
on raw embeddings. This module is the opposite: it is imported by the HOST trainer and computes
L_sep inside the host's own loss graph, before backward, before R6-04's surrogate composition or
R4-08's PCGrad projection, and before the optimizer step.

Everything it needs is the host's own state:
  * ``z`` is the level-0 residual from the host's ``_residuals_with_trace``, i.e. the encoder
    output the quantizer actually consumes -- not a raw item embedding;
  * the L1 distance is the host layer's own ``_distances``;
  * the source-conditioned exclusion uses the host's own ``successor_indicator`` CSR and the
    host's real ``source_ids`` / ``target_ids`` item IDs.

When ``R7_F_ENABLED`` is unset (the default) nothing here is imported into the loss path and the
production run is bit-identical to the host without the hook.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

# The floor the issue's tau rule fixes. A tau below it would make the surrogate's gradient
# vanish, and the issue says that must be a FEASIBILITY FAILURE, not a silent retune.
TAU_FLOOR = 1e-3


class InvalidProtocol(RuntimeError):
    """Raised when the auxiliary loss cannot be applied correctly.

    The issue requires these to be hard errors. A missing training graph, an empty valid pair set
    or a stale step must never be silently treated as "no effect".
    """

    def __init__(self, status: str, reason: str):
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason


def enabled() -> bool:
    return os.environ.get("R7_F_ENABLED", "0") == "1"


def lambda_weight() -> float:
    return float(os.environ.get("R7_F_LAMBDA", "0.10"))


def tau() -> float:
    return float(os.environ.get("R7_F_TAU", str(TAU_FLOOR)))


def candidate_count() -> int:
    return int(os.environ.get("R7_F_CANDIDATES", "128"))


def training_csr(n_items: int):
    """The (n_items, n_items) boolean CSR of true train transitions, built from the parquet.

    The host only constructs its own ``successor_indicator`` for one arm, so for the others this
    builds the identical structure from the SAME training-only source. It is cached per process.
    ``sents`` are the last item in each user's history -> the next observed target.
    """
    global _CSR_CACHE
    if _CSR_CACHE is not None:
        return _CSR_CACHE
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    _root = Path(__file__).resolve().parent.parent
    candidates = [
        _root / "dataset/Amazon_2023_Instruments/train.parquet",
        _root / "results/stage0_build_parquet/train.parquet",
        _root / "stage0_build_parquet/train.parquet",
    ]
    frame = None
    for path in candidates:
        if path.is_file():
            frame = pd.read_parquet(path, columns=["seen_history", "target"])
            break
    if frame is None:
        raise InvalidProtocol("invalid_protocol",
                              "no training parquet is available, so the train-only relationship "
                              "graph cannot be built and every candidate would be unrelated")
    rows: list[int] = []
    cols: list[int] = []
    for history, target in zip(frame["seen_history"].to_numpy(),
                               frame["target"].to_numpy(dtype=np.int64)):
        if history is None or len(history) == 0:
            continue
        rows.append(int(history[-1]))
        cols.append(int(target))
    if not rows:
        raise InvalidProtocol("invalid_protocol", "the training parquet has no transitions")
    size = max(n_items, max(rows + cols) + 1)
    matrix = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(size, size))
    matrix.sum_duplicates()
    matrix.data[:] = True
    _CSR_CACHE = matrix
    return matrix


_CSR_CACHE = None


def _successor_sets(successor_indicator: Any, source_ids: torch.Tensor) -> list[set[int]]:
    """Per-source successor sets from the host's own CSR, indexed by TRUE item id."""
    if successor_indicator is None:
        # Build the identical structure from the SAME training-only source rather than treating
        # every candidate as unrelated. A missing graph is still a hard error inside training_csr.
        successor_indicator = training_csr(int(source_ids.max()) + 1)
    sets: list[set[int]] = []
    for item in source_ids.tolist():
        row = successor_indicator[int(item)]
        try:
            sets.append(set(int(col) for col in row.indices))
        except AttributeError:
            dense = row.toarray().reshape(-1) if hasattr(row, "toarray") else row
            sets.append({index for index, flag in enumerate(dense.tolist()) if flag})
    return sets


def separation_loss(raw_module, z_level0: torch.Tensor, source_ids: torch.Tensor,
                    target_ids: torch.Tensor, successor_indicator: Any, *,
                    step: int, lam: float | None = None, tau_value: float | None = None,
                    rng: torch.Generator | None = None) -> dict[str, Any]:
    """L_sep = lambda * mean over VALID train-only unrelated pairs of <p_i, p_j>.

    ``z_level0`` is the host's level-0 residual: the encoder output the L1 quantizer sees. The
    distance is the host layer's own, the posteriors are softmax(-d/tau) over that distance, and
    the mask is per SOURCE: self, this source's observed target, and this source's known train
    successors are excluded. A successor of a DIFFERENT source is kept.

    Returns the loss plus the counters the issue requires to be logged per step.
    """
    lam = lambda_weight() if lam is None else lam
    tau_value = tau() if tau_value is None else tau_value
    if lam <= 0.0:
        # The zero-lambda no-op control: the same keys, so the log shape does not change between
        # the arms and a missing field can never be mistaken for a missing measurement.
        return {"loss": None, "valid_count": 0, "candidate_count": 0, "empty_count": 0,
                "mean_p_same": None, "max_p_same": None, "tau": tau_value, "lambda": lam}
    if tau_value < TAU_FLOOR:
        raise InvalidProtocol("invalid_protocol",
                              f"tau={tau_value} is below the stability floor {TAU_FLOOR}")
    layer = raw_module.rq.vq_layers[0]
    pairs = int(source_ids.shape[0])
    # The level-0 residual of the behaviour pair batch is the CONCATENATION [source; target], so
    # its row count is twice the pair count. The SOURCE half carries the item ids in source_ids,
    # and candidates are drawn among those source instances so that every comparison is between
    # real item ids rather than between pair-batch positions.
    if z_level0.shape[0] == 2 * pairs:
        z_level0 = z_level0[:pairs]
    elif z_level0.shape[0] != pairs:
        raise InvalidProtocol("invalid_protocol",
                              f"the latent batch ({z_level0.shape[0]}) is neither the pair count "
                              f"({pairs}) nor twice it, so the source half cannot be identified")
    n_items = int(z_level0.shape[0])
    generator = rng or torch.Generator().manual_seed(int(step))
    count = candidate_count()
    left = torch.randint(0, n_items, (count,), generator=generator)
    right = (left + torch.randint(1, max(2, n_items), (count,), generator=generator)) % n_items
    candidates = torch.stack([left, right], dim=1)

    successors = _successor_sets(successor_indicator, source_ids)
    keep = torch.ones(candidates.shape[0], dtype=torch.bool)
    for row in range(candidates.shape[0]):
        position_i = int(candidates[row, 0])
        position_j = int(candidates[row, 1])
        if position_i == position_j:
            keep[row] = False
            continue
        # The exclusion is per SOURCE: item_j may not be this source's observed target, and may
        # not be this source's known train successor. A successor of a DIFFERENT source is kept.
        item_j = int(source_ids[position_j])
        if int(target_ids[position_i]) == item_j:
            keep[row] = False
            continue
        if item_j in successors[position_i]:
            keep[row] = False
    valid = candidates[keep]
    if valid.shape[0] == 0:
        raise InvalidProtocol("invalid_protocol",
                              "the exclusion mask removed every candidate pair, so L_sep has no "
                              "term to minimise; this is a feasibility failure, not a zero loss")
    distance = layer._distances(z_level0.reshape(-1, layer.dim))
    posterior = torch.softmax(-distance / tau_value, dim=-1)
    overlap = (posterior[valid[:, 0]] * posterior[valid[:, 1]]).sum(dim=-1)
    loss = lam * overlap.mean()
    return {
        "loss": loss,
        "valid_count": int(valid.shape[0]),
        "candidate_count": int(candidates.shape[0]),
        "empty_count": int((~keep).sum()),
        "mean_p_same": float(overlap.detach().mean()),
        "max_p_same": float(overlap.detach().max()),
        "tau": tau_value,
        "lambda": lam,
    }


def _position_of(source_ids: torch.Tensor, item: int) -> int | None:
    matches = torch.nonzero(source_ids == int(item), as_tuple=False)
    return int(matches.reshape(-1)[0]) if matches.numel() else None


def native_gap_quantiles(raw_module, z_level0: torch.Tensor) -> dict[str, float]:
    """p05 / median / p95 of the NEAREST vs RUNNER-UP L1 distance gap, on a REAL latent.

    This is the quantity the issue's tau rule asks for, computed on the encoder's own output with
    the quantizer's own distance. The #87 pilot reported a median of 0 because it measured the
    wrong tensor; that is the defect this function exists to make impossible to repeat.
    """
    layer = raw_module.rq.vq_layers[0]
    with torch.no_grad():
        distance = layer._distances(z_level0.reshape(-1, layer.dim))
        ordered, _ = distance.sort(dim=1)
        gap = ordered[:, 1] - ordered[:, 0]
    return {
        "p05": float(gap.quantile(0.05)),
        "median": float(gap.median()),
        "p95": float(gap.quantile(0.95)),
        "mean": float(gap.mean()),
        "items": int(gap.numel()),
    }


def hard_l1_indices(raw_module, embeddings: torch.Tensor) -> torch.Tensor:
    """The host's NATIVE hard L1 assignment for the given item embeddings.

    The #87 pilot took argmax over a soft posterior built on raw embeddings; this is the canonical
    hard identity the issue requires, obtained through the model's own inference path.
    """
    layer = raw_module.rq.vq_layers[0]
    with torch.no_grad():
        trace = raw_module.encoder(embeddings)
        if isinstance(trace, tuple):
            trace = trace[0]
        distance = layer._distances(trace.reshape(-1, layer.dim))
        return distance.argmin(dim=-1)


# --------------------------------------------------------------------------------------
# provenance helpers used by the host trainer's per-step log
# --------------------------------------------------------------------------------------
def sha256_tensor(tensor: torch.Tensor) -> str:
    import hashlib
    with torch.no_grad():
        value = tensor.detach().to(torch.float64).cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def model_digest(model) -> str:
    """A content hash of the model's parameters, so a step's state is identifiable."""
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        with torch.no_grad():
            payload = value.detach().to(torch.float64).cpu().contiguous()
        digest.update(hashlib.sha256(payload.numpy().tobytes()).digest())
    return digest.hexdigest()


def grad_norm(loss: torch.Tensor | None, parameters: list, model, *, retain: bool = True) -> float | None:
    """The L2 norm of the SEPARATION term's gradient on the given parameters.

    A missing gradient is reported as None, never as 0.0: "disconnected" and "zero" are different
    facts and the issue requires them to be distinguished.
    """
    if loss is None or not parameters:
        return None
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        total += float((grad.detach().to(torch.float64) ** 2).sum())
    return total ** 0.5


def save_snapshot(model, optimizer, scheduler, step: int, directory: str,
                  *, label: str | None = None) -> str:
    """Persist the model/optimizer/scheduler at a declared step."""
    from pathlib import Path
    import torch
    target = Path(directory) / f"step_{step:04d}"
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "label": label,
        "model_state": {name: value.detach().clone().cpu()
                        for name, value in model.state_dict().items()},
        "optimizer_state": {key: value for key, value in optimizer.state_dict().items()},
        "scheduler_state": (scheduler.state_dict() if scheduler is not None else None),
    }
    torch.save(payload, target / "snapshot.pt")
    return str(target / "snapshot.pt")


# --------------------------------------------------------------------------------------
# R8 ten-mechanism dispatch (issues #91..#100)
# --------------------------------------------------------------------------------------
def r8_lane() -> str:
    return os.environ.get("R8_LANE", "")


def r8_loss(raw_module, residuals, source_ids, target_ids, successor_indicator, *, step: int):
    """Run the configured R8 lane's auxiliary term through tools/r8_mechanisms.py."""
    import r8_mechanisms
    return r8_mechanisms.compute(raw_module, residuals, source_ids, target_ids,
                                 successor_indicator, step=step)
