"""Side-effect-free gradient / clipping / AdamW trajectory recorder (issue #240).

Design contract
---------------
The recorder is a pure observer of one training step.  It

* reads the per-level quantisation terms through forward hooks on the unchanged
  ``VQLayer`` modules (no module code is patched, no value is replaced),
* reads the per-level behaviour terms through ``_BEHAVIOR_LEVEL_LIST``, which the
  trainer appends to but never reads back,
* probes each loss term's gradient with ``torch.autograd.grad``.  That call does
  not create ``AccumulateGrad`` nodes, so ``.grad`` is untouched and DDP's
  post-accumulate hooks never fire: the real ``loss.backward()`` afterwards sees
  the same graph and produces the same gradients as an un-instrumented run,
* snapshots parameters before ``optimizer.step()`` and diffs them after, which
  observes AdamW without altering it.

Everything is written from rank 0.  ``world_size == 1`` for the frozen resample
recipe, so rank 0's gradients *are* the global gradients; the recorder still
writes ``world_size`` and ``rank`` into every row so the claim stays auditable.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

# === test-only switch: the equivalence harness flips this single line ===
DIAG_ENABLED = True

_HERE = Path(__file__).resolve()
_REPO = None
for _parent in _HERE.parents:
    if (_parent / "stage2_RQ-VAE").is_dir():
        _REPO = _parent
        break
if _REPO is None:  # pragma: no cover - the tree always sits under the repo
    raise RuntimeError("cannot locate the repository root from " + str(_HERE))

OUT_DIR = _REPO / "results" / "research" / "issue240"
RAW_DIR = OUT_DIR / "raw"
PROBE_SEED = 20240101
PROBE_ITEMS = 2048


def _f(value):
    """JSON-safe scalar: NaN/Inf become None, strings and booleans pass through."""
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def param_groups(module) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {
        "encoder": list(module.encoder.parameters()),
        "decoder": list(module.decoder.parameters()),
    }
    for index, layer in enumerate(module.rq.vq_layers, start=1):
        groups[f"codebook_l{index}"] = [layer.embed.weight]
    groups["all"] = list(module.parameters())
    return groups


def _stats(tensors) -> dict:
    """L2 / RMS / max-abs / non-zero fraction, with None separated from true zeros."""
    total_params = 0
    none_tensors = 0
    zero_tensors = 0
    nonzero_elements = 0
    sum_sq = 0.0
    max_abs = 0.0
    seen = 0
    for tensor in tensors:
        if tensor is None:
            none_tensors += 1
            continue
        seen += 1
        flat = tensor.detach()
        total_params += flat.numel()
        if flat.numel() == 0:
            zero_tensors += 1
            continue
        sum_sq += float(flat.pow(2).sum())
        max_abs = max(max_abs, float(flat.abs().max()))
        nonzero_elements += int((flat != 0).sum())
        if not bool((flat != 0).any()):
            zero_tensors += 1
    l2 = math.sqrt(sum_sq)
    return {
        "n_tensors": seen,
        "n_none": none_tensors,
        "n_zero_tensors": zero_tensors,
        "n_params": total_params,
        "l2": l2,
        "rms": (math.sqrt(sum_sq / total_params) if total_params else 0.0),
        "max_abs": max_abs,
        "nonzero_frac": (nonzero_elements / total_params if total_params else 0.0),
        "all_none": seen == 0,
    }


def _flat(grads, params) -> torch.Tensor | None:
    """Canonical flattening: ``None`` is a genuine zero for that parameter.

    ``autograd.grad(..., allow_unused=True)`` returns ``None`` for a parameter the
    term does not touch.  Concatenating only the non-``None`` entries would give
    every term a different length, so the layout is fixed by the parameter order
    and missing entries become zeros of the right shape.
    """
    if grads is None:
        return None
    parts = [
        (p.new_zeros(p.numel()) if g is None else g.reshape(-1))
        for g, p in zip(grads, params)
    ]
    return torch.cat(parts) if parts else None


def _direction(left: torch.Tensor | None, right: torch.Tensor | None) -> dict:
    if left is None or right is None:
        return {"cos": None, "dot": None, "na": True, "left_norm": None, "right_norm": None}
    left_norm = float(left.norm())
    right_norm = float(right.norm())
    if left_norm == 0.0 or right_norm == 0.0:
        return {
            "cos": None, "dot": float(torch.dot(left, right)), "na": True,
            "left_norm": left_norm, "right_norm": right_norm,
        }
    dot = float(torch.dot(left, right))
    return {
        "cos": dot / (left_norm * right_norm), "dot": dot, "na": False,
        "left_norm": left_norm, "right_norm": right_norm,
    }


def _ratio(numerator: float | None, denominator: float | None) -> dict:
    """A ratio with an explicit guard: near-zero denominators are marked, not divided."""
    if numerator is None or denominator is None:
        return {"value": None, "guard": "missing"}
    if denominator < 1e-12:
        return {"value": None, "guard": "denominator_below_1e-12", "denominator": denominator}
    return {"value": numerator / denominator, "guard": None, "denominator": denominator}


class Recorder:
    """Per-step observer.  ``enabled=False`` makes every entry point a no-op."""

    def __init__(self, module, device, rank: int, world_size: int, enabled: bool = True):
        self.enabled = bool(enabled) and DIAG_ENABLED
        self.module = module
        self.device = device
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.groups = param_groups(module) if self.enabled else {}
        self.params = self.groups.get("all", []) if self.enabled else []
        self.quant_calls: list[tuple] = []
        self._index = ({id(p): i for i, p in enumerate(self.params)} if self.enabled else {})
        self.phase: str | None = None
        self.step_index = 0
        self.epoch = 0
        self.rows = 0
        self._probe_grads: dict[str, tuple] = {}
        self._snapshot: dict[str, list[torch.Tensor]] = {}
        self._handles = []
        self._fh = None
        self.prev_probe: dict | None = None
        self.probe_indices = None
        self.overhead_s = 0.0
        self.probe_budget = []
        if self.enabled:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            if self.rank == 0:
                self._fh = open(RAW_DIR / "steps.jsonl", "w", encoding="utf-8")

    # ------------------------------------------------------------------ hooks
    def install(self) -> None:
        if not self.enabled:
            return
        for layer in self.module.rq.vq_layers:
            self._handles.append(layer.register_forward_hook(self._quant_hook))

    def _quant_hook(self, module, inputs, output):
        if not self.enabled or self.phase is None:
            return
        x = inputs[0]
        x_q, quant_loss, _unused, tokens = output
        self.quant_calls.append((self.phase, x, x_q, quant_loss, tokens))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------------ per step
    def begin_step(self, epoch: int, step: int) -> None:
        if not self.enabled:
            return
        self.epoch = int(epoch)
        self.step_index = int(step)
        self.quant_calls = []
        self.phase = None
        self._probe_grads = {}
        self._t0 = time.time()

    def probe(self, *, base_loss, recon_loss, behavior_levels, behavior_weight: float, beta: float) -> None:
        """Probe per-term gradients.  Must run before ``loss.backward()``."""
        if not self.enabled:
            return
        main_calls = [call for call in self.quant_calls if call[0] == "main"]
        if len(main_calls) != 3:
            raise RuntimeError(f"expected 3 main quantiser calls, saw {len(main_calls)}")
        terms: dict[str, torch.Tensor] = {"recon": recon_loss}
        for level, (_phase, x, _x_q_ste, _ql, tokens) in enumerate(main_calls):
            # VQLayer.forward returns the straight-through tensor x + (x_q - x).detach(),
            # whose value equals x, so rebuilding x_q from the emitted tokens with the
            # same codebook parameter is the only way to recover the pre-STE value that
            # quant_loss was actually computed from.  Same indices, same weight, same
            # graph edges, so the gradient is the one the real backward sees.
            weight = self.module.rq.vq_layers[level].get_code_embs()
            x_q = torch.nn.functional.embedding(tokens, weight)
            # RQLayer.forward returns sum_quant_loss / codebook_num, so each level's
            # (codebook + beta * commitment) enters the loss with coefficient 1/3.
            # Terms stay unweighted here; the coefficient lives in ``_coef``.
            terms[f"cb_l{level + 1}"] = torch.nn.functional.mse_loss(x_q, x.detach())
            terms[f"commit_l{level + 1}"] = torch.nn.functional.mse_loss(x, x_q.detach())
        for level, loss in enumerate(behavior_levels):
            terms[f"behavior_l{level + 1}"] = loss
        levels = len(main_calls) or 1
        self._coef = {
            "recon": 1.0,
            **{f"cb_l{i}": 1.0 / levels for i in (1, 2, 3)},
            **{f"commit_l{i}": beta / levels for i in (1, 2, 3)},
            **{f"behavior_l{i}": behavior_weight / max(len(behavior_levels), 1) for i in (1, 2, 3)},
        }
        grads: dict[str, tuple] = {}
        for name, term in terms.items():
            grads[name] = torch.autograd.grad(
                term, self.params, retain_graph=True, allow_unused=True,
            )
        self._probe_grads = grads
        self._base_loss_value = float(base_loss.detach())
        self._behavior_weight = float(behavior_weight)
        self._term_values = {name: float(term.detach()) for name, term in terms.items()}
        self._t_probe = time.time()

    def after_backward(self) -> None:
        """Called after the real ``loss.backward()``, before clipping."""
        if not self.enabled:
            return
        self._grad_before_clip = [None if p.grad is None else p.grad.detach().clone()
                                  for p in self.params]
        self._norm_before_clip = math.sqrt(sum(
            float(g.pow(2).sum()) for g in self._grad_before_clip if g is not None
        ))
        # identity check: sum of weighted term gradients == the real backward gradient
        predicted: dict[str, torch.Tensor | None] = {}
        for group_name, params in self.groups.items():
            predicted[group_name] = None
        if self._probe_grads:
            acc = None
            for name, grads in self._probe_grads.items():
                coef = self._coef[name]
                contribution = [None if g is None else g * coef for g in grads]
                if acc is None:
                    acc = contribution
                else:
                    acc = [
                        (a if b is None else b) if (a is None or b is None) else a + b
                        for a, b in zip(acc, contribution)
                    ]
            flat_pred = _flat(acc, self.params) if acc is not None else None
            flat_actual = _flat(self._grad_before_clip, self.params)
            identity = _direction(flat_pred, flat_actual)
            if flat_pred is not None and flat_actual is not None:
                identity["rel_error"] = float((flat_pred - flat_actual).norm()) / max(float(flat_actual.norm()), 1e-12)
            self._identity = identity
            self._predicted_stats = _stats(acc if acc is not None else [])
        else:
            self._identity = {"cos": None, "dot": None, "na": True}
            self._predicted_stats = {}

    def after_clip(self, clip_total_norm) -> None:
        if not self.enabled:
            return
        self._clip_total_norm = None if clip_total_norm is None else float(clip_total_norm)
        self._grad_after_clip = [None if p.grad is None else p.grad.detach().clone()
                                 for p in self.params]
        self._norm_after_clip = math.sqrt(sum(
            float(g.pow(2).sum()) for g in self._grad_after_clip if g is not None
        ))

    def before_step(self) -> None:
        if not self.enabled:
            return
        self._snapshot = {
            name: [p.detach().clone() for p in params]
            for name, params in self.groups.items() if name != "all"
        }

    def after_step(self, lr: float) -> None:
        if not self.enabled:
            return
        deltas = {}
        for name, params in self.groups.items():
            if name == "all":
                continue
            before = self._snapshot.get(name, [])
            l2 = 0.0
            ref = 0.0
            max_abs = 0.0
            n = 0
            for p, old in zip(params, before):
                diff = p.detach() - old.to(p.device)
                l2 += float(diff.pow(2).sum())
                ref += float(old.pow(2).sum())
                max_abs = max(max_abs, float(diff.abs().max()) if diff.numel() else 0.0)
                n += diff.numel()
            deltas[name] = {
                "l2": math.sqrt(l2),
                "rms": math.sqrt(l2 / n) if n else 0.0,
                "max_abs": max_abs,
                "rel": (math.sqrt(l2) / math.sqrt(ref)) if ref > 0 else None,
                "n_params": n,
            }
        self._deltas = deltas
        self._lr = float(lr)
        self._write_row()
        self.overhead_s += time.time() - self._t0

    # ------------------------------------------------------------------ output
    def _per_module(self, grads) -> dict:
        out = {}
        for group_name, params in self.groups.items():
            if group_name == "all":
                continue
            # the probe tuple is aligned with ``self.params``; slice it once per group
            selected = [grads[self._index[id(p)]] if id(p) in self._index else None for p in params]
            out[group_name] = _stats(selected)
        return out

    def _write_row(self) -> None:
        if self.rank != 0 or self._fh is None:
            return
        term_stats = {}
        for name, grads in self._probe_grads.items():
            stats = _stats(grads)
            stats["weighted_l2"] = stats["l2"] * abs(self._coef[name])
            term_stats[name] = stats
        flat = {name: _flat(grads, self.params) for name, grads in self._probe_grads.items()}
        quant_total = None
        for name in ("cb_l1", "cb_l2", "cb_l3", "commit_l1", "commit_l2", "commit_l3"):
            vec = flat.get(name)
            if vec is None:
                continue
            quant_total = vec if quant_total is None else quant_total + vec
        behavior_total = None
        for name in ("behavior_l1", "behavior_l2", "behavior_l3"):
            vec = flat.get(name)
            if vec is None:
                continue
            behavior_total = vec if behavior_total is None else behavior_total + vec
        if behavior_total is not None:
            behavior_total = behavior_total / 3.0  # behavior_contrastive_loss averages 3 levels
        commit_total = None
        for name in ("commit_l1", "commit_l2", "commit_l3"):
            vec = flat.get(name)
            if vec is None:
                continue
            commit_total = vec if commit_total is None else commit_total + vec
        base_vec = flat.get("recon")
        if base_vec is not None and quant_total is not None:
            base_vec = base_vec + quant_total
        directions = {
            "recon_vs_behavior": _direction(flat.get("recon"), behavior_total),
            "base_vs_behavior": _direction(base_vec, behavior_total),
            "quant_vs_behavior": _direction(quant_total, behavior_total),
            "commit_vs_behavior": _direction(commit_total, behavior_total),
            "recon_vs_quant": _direction(flat.get("recon"), quant_total),
            "behavior_l1_vs_l2": _direction(flat.get("behavior_l1"), flat.get("behavior_l2")),
            "behavior_l1_vs_l3": _direction(flat.get("behavior_l1"), flat.get("behavior_l3")),
            "behavior_l2_vs_l3": _direction(flat.get("behavior_l2"), flat.get("behavior_l3")),
        }
        for key, value in directions.items():
            for field, field_value in value.items():
                directions[key][field] = _f(field_value)
        behavior_norm = float(behavior_total.norm()) if behavior_total is not None else None
        base_norm = float(base_vec.norm()) if base_vec is not None else None
        ratios = {
            "unweighted_behavior_over_recon": _ratio(
                behavior_norm, term_stats.get("recon", {}).get("l2")
            ),
            "weighted_behavior_over_recon": _ratio(
                (behavior_norm * self._behavior_weight) if behavior_norm is not None else None,
                term_stats.get("recon", {}).get("l2"),
            ),
            "weighted_behavior_over_base": _ratio(
                (behavior_norm * self._behavior_weight) if behavior_norm is not None else None,
                base_norm,
            ),
            "quant_over_recon": _ratio(
                float(quant_total.norm()) if quant_total is not None else None,
                term_stats.get("recon", {}).get("l2"),
            ),
            "behavior_abs_norm": {"value": behavior_norm, "guard": None},
            "base_abs_norm": {"value": base_norm, "guard": None},
        }
        row = {
            "epoch": self.epoch,
            "step": self.step_index,
            "rank": self.rank,
            "world_size": self.world_size,
            "lr": self._lr,
            "behavior_weight": self._behavior_weight,
            "losses": {
                **{name: _f(value) for name, value in self._term_values.items()},
                "base_total": _f(self._base_loss_value),
                "weighted_behavior": _f(
                    self._term_values.get("behavior_l1", 0.0) * self._behavior_weight
                ),
            },
            "term_stats": term_stats,
            "per_module": {name: self._per_module(grads)
                           for name, grads in self._probe_grads.items()},
            "directions": directions,
            "ratios": {name: {k: _f(v) for k, v in value.items()} for name, value in ratios.items()},
            "clip": {
                "pre_norm_observed": _f(self._norm_before_clip),
                "pre_norm_reported": _f(getattr(self, "_clip_total_norm", None)),
                "post_norm": _f(self._norm_after_clip),
                "coef": _f(min(1.0, 1.0 / (self._clip_total_norm + 1e-6))
                           if getattr(self, "_clip_total_norm", None) else None),
                "triggered": bool(getattr(self, "_clip_total_norm", 0.0) > 1.0),
            },
            "identity": {k: _f(v) for k, v in self._identity.items()},
            "adam_delta": self._deltas,
        }
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        self.rows += 1

    # ------------------------------------------------------------------ per epoch
    def end_epoch(self, epoch: int) -> None:
        if not self.enabled:
            return
        embeddings = getattr(self, "_all_embeddings", None)
        if embeddings is None:
            return
        if self.probe_indices is None:
            generator = torch.Generator().manual_seed(PROBE_SEED)
            self.probe_indices = torch.randperm(len(embeddings), generator=generator)[:PROBE_ITEMS]
        probe = embeddings[self.probe_indices].to(self.device)
        layers = self.module.rq.vq_layers
        previous = [getattr(layer, "_skip_ddp_reduce", False) for layer in layers]
        for layer in layers:
            layer._skip_ddp_reduce = True
        try:
            with torch.no_grad():
                z = self.module.encoder(probe)
                residual = z
                residuals = []
                tokens = []
                for layer in layers:
                    residuals.append(residual.detach().clone())
                    _x_q, _ql, _unused, index = layer(residual, infer_use_sk=False)
                    residual = residual - _x_q
                    tokens.append(index.reshape(-1))
                hard_q = sum(
                    torch.nn.functional.embedding(index, layer.get_code_embs())
                    for index, layer in zip(tokens, layers)
                )
        finally:
            for layer, flag in zip(layers, previous):
                layer._skip_ddp_reduce = flag
        current = {
            "z": z.detach().cpu(),
            "residuals": [r.cpu() for r in residuals],
            "hard_q": hard_q.detach().cpu(),
            "tokens": torch.stack(tokens, dim=1).cpu(),
        }
        record = {"epoch": int(epoch)}
        if self.prev_probe is not None:
            record["z_shift"] = float((current["z"] - self.prev_probe["z"]).norm())
            record["hard_q_shift"] = float((current["hard_q"] - self.prev_probe["hard_q"]).norm())
            record["residual_shift"] = [
                float((a - b).norm()) for a, b in zip(current["residuals"], self.prev_probe["residuals"])
            ]
            record["assignment_change"] = float(
                (current["tokens"] != self.prev_probe["tokens"]).any(dim=1).float().mean()
            )
            record["per_level_change"] = [
                float((current["tokens"][:, i] != self.prev_probe["tokens"][:, i]).float().mean())
                for i in range(current["tokens"].shape[1])
            ]
        codebooks = {}
        for level, layer in enumerate(layers, start=1):
            weight = layer.get_code_embs().detach()
            gram = torch.cdist(weight, weight)
            gram.fill_diagonal_(float("inf"))
            nearest = gram.min(dim=1).values
            counts = torch.bincount(current["tokens"][:, level - 1], minlength=weight.shape[0]).float()
            probs = counts / counts.sum()
            entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum())
            codebooks[f"l{level}"] = {
                "used": int((counts > 0).sum()),
                "entropy": entropy,
                "max_share": float(probs.max()),
                "spacing_min": float(nearest.min()),
                "spacing_mean": float(nearest.mean()),
                "codeword_norm_mean": float(weight.norm(dim=1).mean()),
            }
        record["codebooks"] = codebooks
        self.prev_probe = current
        with open(OUT_DIR / "epochs.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def attach_embeddings(self, embeddings) -> None:
        if self.enabled:
            self._all_embeddings = embeddings

    def finalize(self, **extra) -> None:
        """Close the stream and write the run manifest (rank 0 only)."""
        if not self.enabled:
            return
        self.close()
        if self.rank != 0:
            return
        manifest = {
            "enabled": True,
            "rank": self.rank,
            "world_size": self.world_size,
            "steps_recorded": self.rows,
            "overhead_s": self.overhead_s,
            "probe_seed": PROBE_SEED,
            "probe_items": PROBE_ITEMS,
            "param_groups": {
                name: int(sum(p.numel() for p in params))
                for name, params in self.groups.items()
            },
            "raw": str(RAW_DIR / "steps.jsonl"),
            **extra,
        }
        (OUT_DIR / "diag_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
