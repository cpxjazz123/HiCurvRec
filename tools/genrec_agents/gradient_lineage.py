"""The gradient lineage, measured on the host's real assembly path (issue #106, WS-A).

A mechanism's gradient is not a thing that either arrives or does not: it is assembled,
projected, clipped and applied, and a mechanism that "reaches the codebook" in isolation
may still be overwritten before the optimizer sees it. This module traces that path
**through the host's own functions** -- the host's backward, the host's conflict-projected
assembly, the host's clipping and the host's optimizer step -- and reports, per parameter
group (the encoder, every RQ codebook level, the decoder), what happened at each stage.

It measures two things at once: the lineage of the host's own objective, and the lineage of
one mechanism's contribution, obtained by running the same case with and without the
mechanism and differencing. The difference is what the mechanism actually delivered.

Nothing here trains: it is a fixture-sized forward and backward through the real code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

EMBEDDINGS = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_trainer(tree: Path):
    for entry in (str(Path(tree).resolve()), str(REPO_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return _load_module(Path(tree) / "train_rqvae.py", "genrec_lineage_trainer")


def _groups(model) -> Dict[str, List[Any]]:
    """The parameter groups the lineage is reported for."""
    groups: Dict[str, List[Any]] = {"encoder": list(model.encoder.parameters())}
    for index, layer in enumerate(getattr(model.rq, "vq_layers", [])):
        groups[f"rq_codebook_{index}"] = list(layer.parameters())
    groups["decoder"] = list(getattr(model, "decoder", torch_nn_placeholder()).parameters()) \
        if hasattr(model, "decoder") else []
    return groups


def torch_nn_placeholder():
    import torch.nn as nn

    return nn.Identity()


def _norms(groups: Dict[str, List[Any]]) -> Dict[str, float]:
    import torch

    out: Dict[str, float] = {}
    for name, parameters in groups.items():
        if not parameters:
            out[name] = 0.0
            continue
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total = max(total, float(parameter.grad.detach().abs().max()))
        out[name] = total
    return out


def _snapshot(groups: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    return {
        name: [parameter.detach().clone() for parameter in parameters]
        for name, parameters in groups.items()
    }


def _displacement(before: Dict[str, List[Any]], after: Dict[str, List[Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, parameters in before.items():
        if not parameters:
            out[name] = 0.0
            continue
        out[name] = max(
            float((old - new).abs().max()) for old, new in zip(parameters, after[name])
        )
    return out


def trace(
    tree: Path,
    *,
    mechanism_module: Optional[str] = None,
    mechanism_weight: float = 0.01,
    mechanism_parameters: Optional[Dict[str, Any]] = None,
    sample: int = 512,
    seed: int = 42,
) -> Dict[str, Any]:
    """Trace the gradient lineage of the host's objective, with and without a mechanism."""
    import torch

    trainer = _load_trainer(Path(tree))
    embeddings = np_load(EMBEDDINGS)[: 2 * sample]
    torch.manual_seed(seed)
    tensor = torch.from_numpy(embeddings)
    pairs = sample
    module = None
    if mechanism_module:
        module = _load_module(Path(tree) / f"{mechanism_module}.py", "genrec_lineage_mechanism")

    def build_model():
        # The two runs must differ ONLY by the mechanism, so the initialisation is the
        # same in both: a difference in the weights would be read as a gradient effect.
        torch.manual_seed(seed)
        import numpy as _np

        _np.random.seed(seed)
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(embeddings.shape[1]))
        model.eval()
        return model

    def one_run(with_mechanism: bool) -> Dict[str, Any]:
        model = build_model()
        groups = _groups(model)
        reconstructed, quant_loss, _, _ = model.forward(tensor)
        base_loss, _recon = model.compute_loss(tensor, reconstructed, quant_loss)
        residuals = trainer._residuals_with_trace(model, tensor)
        source_ids = torch.arange(pairs)
        target_ids = (source_ids + 1) % pairs
        l1_loss, other_loss, valid_levels = trainer.behavior_contrastive_components(
            residuals, source_ids, target_ids, None
        )
        loss = base_loss + l1_loss
        mechanism_value = None
        if with_mechanism and module is not None:
            source = residuals[0, :pairs]
            target = residuals[0, pairs : 2 * pairs]
            codebook = model.rq.vq_layers[0].get_code_embs()
            mask = torch.ones(pairs, dtype=source.dtype)
            parameters = dict(mechanism_parameters or {})
            mechanism_value = module.mechanism_loss(source, target, codebook, mask, **parameters)
            loss = loss + mechanism_weight * mechanism_value

        model.zero_grad(set_to_none=False)
        loss.backward(retain_graph=True)
        after_backward = _norms(groups)

        # the host's own assembly: it overwrites the encoder gradient
        if getattr(trainer, "R4_08_PCGRAD_ENC", False):
            trainer._r4_08_apply_projection(
                base_loss,
                l1_loss,
                other_loss,
                list(model.encoder.parameters()),
                1.0,
                valid_levels=valid_levels,
            )
        after_assembly = _norms(groups)

        clip = float(getattr(trainer, "GRADIENT_CLIP_NORM", 0.0) or 0.0)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        after_clip = _norms(groups)

        before = _snapshot(groups)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer.step()
        after = _snapshot(groups)
        displacement = _displacement(before, after)
        return {
            "after_backward": after_backward,
            "after_host_assembly": after_assembly,
            "after_clipping": after_clip,
            "after_optimizer_step": displacement,
            "mechanism_value": None if mechanism_value is None else float(mechanism_value.detach()),
        }

    host = one_run(False)
    with_mechanism = one_run(True) if module is not None else None

    report: Dict[str, Any] = {
        "tree": str(tree),
        "mechanism_module": mechanism_module,
        "sample_items": pairs,
        "seed": seed,
        "host_objective": host,
        "host_assembly_overwrites_encoder": bool(getattr(trainer, "R4_08_PCGRAD_ENC", False)),
    }
    if with_mechanism is not None:
        delivered: Dict[str, Dict[str, float]] = {}
        for stage in ("after_backward", "after_host_assembly", "after_clipping"):
            delivered[stage] = {
                name: with_mechanism[stage][name] - host[stage][name]
                for name in host[stage]
            }
        report["mechanism_contribution"] = {
            "by_stage": delivered,
            "after_optimizer_step": {
                name: with_mechanism["after_optimizer_step"][name]
                - host["after_optimizer_step"][name]
                for name in host["after_optimizer_step"]
            },
            "delivered_to_encoder": delivered["after_host_assembly"].get("encoder", 0.0) > 0.0,
            "delivered_to_codebooks": {
                name: value
                for name, value in delivered["after_host_assembly"].items()
                if name.startswith("rq_codebook") and value > 0.0
            },
        }
    return report


def np_load(path: Path):
    import numpy as np

    return np.load(path).astype("float32")


__all__ = ["trace"]
