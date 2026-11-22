"""The encoder's gradient, measured on the candidate's own model.

The three implementations of the frozen specification may agree perfectly and
the real encoder can still receive nothing: an accidental ``detach`` in the
wiring, a quantization path that skips the straight-through estimator, or a
mechanism whose output the loss never reaches. This probe loads the candidate's
model, runs one forward and one backward on a batch of embeddings, and reports
whether the ENCODER's parameters actually received a gradient.

Run as ``python -m tools.research.stage2.gradient_probe --tree <tree>``; one JSON
line on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _call_model(model, embeddings, ids):
    """Call the candidate's model, whatever its forward takes.

    A candidate's model may take the item ids beside the embeddings and may
    return the reconstruction's reference beside the other four values; the
    frozen baseline's model takes the embeddings alone and returns four. The
    probe asks one question - does the reconstruction reach the encoder - so it
    adapts to the interface instead of demanding one.
    """
    import inspect

    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        signature = None
    # A DDP wrapper's forward takes *inputs, so the POSITIONAL parameters are
    # counted, not the names: only a model that declares a second positional
    # argument receives the ids.
    positional = []
    if signature is not None:
        positional = [parameter for parameter in signature.parameters.values()
                      if parameter.kind in (parameter.POSITIONAL_ONLY,
                                            parameter.POSITIONAL_OR_KEYWORD)]
    call = model if callable(model) else model.forward
    result = call(embeddings, ids) if len(positional) > 1 else call(embeddings)
    if isinstance(result, (tuple, list)):
        values = list(result)
    else:
        values = [result]
    reference = values[4] if len(values) >= 5 else embeddings
    return values, reference

def probe(tree: Path, *, batch: int = 64, seed: int = 7) -> dict:
    import torch

    tree = Path(tree).resolve()
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    from model import RQVAE  # noqa: E402  (the candidate's own package)

    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_path = tree / "train_rqvae.py"
    spec = {}
    for name in ("HIDDEN_SIZES", "CODEBOOK_NUM", "CODEBOOK_SIZE", "CODEBOOK_DIM",
                 "BETA", "VQ_TYPE", "EMA_DECAY", "SK_EPSILON", "SK_ITERS",
                 "FIELD_ENABLED", "GEO_ENABLED", "STRAT_ENABLED",
                 "HOLONOMY_ENABLED"):
        value = _constant(config_path, name)
        if value is not None:
            spec[name] = value
    if "HIDDEN_SIZES" not in spec:
        return {"status": "NOT_RUN", "reason": "the trainer states no HIDDEN_SIZES"}
    from types import SimpleNamespace

    sizes = spec["CODEBOOK_SIZE"]
    if isinstance(sizes, int):
        sizes = (sizes,) * int(spec["CODEBOOK_NUM"])
    config = SimpleNamespace(
        hidden_sizes=tuple(spec["HIDDEN_SIZES"]),
        codebook_num=int(spec["CODEBOOK_NUM"]),
        codebook_size=tuple(sizes),
        codebook_dim=int(spec["CODEBOOK_DIM"]),
        dropout=0.0, beta=float(spec["BETA"]), vq_type=str(spec["VQ_TYPE"]),
        ema_decay=float(spec["EMA_DECAY"]), fix_code_embs=False,
        sk_epsilon=float(spec["SK_EPSILON"]), sk_iters=int(spec["SK_ITERS"]),
        field_enabled=bool(spec.get("FIELD_ENABLED") or 0),
        geocap_enabled=bool(spec.get("GEO_ENABLED") or 0),
        stratified_enabled=bool(spec.get("STRAT_ENABLED") or 0),
        holonomy_enabled=bool(spec.get("HOLONOMY_ENABLED") or 0),
    )
    dimension = 768
    model = RQVAE(config, in_dim=dimension).to(device)
    embeddings = torch.randn(batch, dimension, device=device) * 0.05
    ids = torch.arange(batch, device=device)
    values, reference = _call_model(model, embeddings, ids)
    reconstructed = values[0]
    # ONLY the reconstruction: the quantisation loss would reach the encoder
    # through the commitment term even when the straight-through estimator is
    # missing, and that is exactly the failure this probe must see.
    loss = torch.nn.functional.mse_loss(reconstructed, reference)
    if not torch.isfinite(loss):
        return {"status": "FAIL", "reason": "the probe loss is not finite"}
    loss.backward()
    encoder = [parameter for name, parameter in model.named_parameters()
               if name.startswith("encoder")]
    if not encoder:
        return {"status": "NOT_RUN", "reason": "the model has no encoder parameters"}
    norms = [float(parameter.grad.abs().sum()) if parameter.grad is not None else 0.0
             for parameter in encoder]
    received = [norm for norm in norms if norm > 0.0]
    detail = "%d of %d encoder tensors received a gradient" % (len(received), len(norms))
    if not received:
        return {"status": "FAIL", "reason": "no encoder parameter received a gradient: "
                                            "the mechanism does not reach the encoder",
                "detail": detail, "norms": norms}
    return {"status": "PASS", "detail": detail,
            "min_nonzero_norm": min(received), "max_norm": max(norms)}


def _constant(path: Path, name: str):
    """The value of a module-level constant, from the trainer's source."""
    import ast

    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except ValueError:
                        return None
    return None


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(probe(Path(arguments.tree)), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
