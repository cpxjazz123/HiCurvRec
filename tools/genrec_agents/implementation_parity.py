"""Does the implementation carry what the frozen proposal declared? (issue #112)

The campaign's sharpest near-miss was a JOIN that passed for a mechanism that was a REDUCED copy
of the approved proposal: the tree implemented the product-geodesic level-0 assignment and nothing
else, while the frozen proposal's distinctive component -- the two factors being SEPARATELY
SUPERVISED -- had been dropped between the freeze and the tree. The contract had been narrowed to
match the implementation, so every gate agreed, and the narrowing was invisible.

**A gate cannot catch a mechanism that is not in the declaration.** The check that would have
caught it compares the IMPLEMENTATION with the PROPOSAL, and this module is that check.

Each mechanism declares the components it is made of, and each component is verified against the
tree:

``symbol_present``
    the named symbol exists in the tree's mechanism module;
``half_scoped_term``
    the named term's gradient reaches one half of the level-0 codebook and leaves the other
    half's share EXACTLY zero -- the role separation, measured rather than asserted.

The result is a PASS/FAIL with the missing components named, so a join can require it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

CGV_DIR = Path(__file__).resolve().parents[1] / "cgv"

#: The components each mechanism is made of, and how each is checked.
COMPONENTS: dict[str, list[dict[str, Any]]] = {
    "hyp_product_l1_quantiser": [
        {"id": "product_assignment", "kind": "symbol_present",
         "symbol": "install_product_assignment",
         "why": "the level-0 assignment is the product-geodesic one"},
        {"id": "role_supervision", "kind": "active_mechanism",
         "symbol": "factor_nce",
         "why": "the behaviour factor's transition term must be the arm's ACTIVE mechanism, not "
                "a function that happens to exist in the tree"},
        {"id": "role_separation", "kind": "half_scoped_term",
         "symbol": "factor_nce",
         "why": "the behaviour factor's transition term must not move the content half"},
    ],
}


def _load_module(tree: Path, relative: str, name: str):
    # the arm's own modules live at the tree's root, and the mechanism module imports them by
    # their bare names, so the root has to be importable
    root = str(tree)
    if root not in sys.path:
        sys.path.insert(0, root)
    path = tree / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _symbol_present(tree: Path, component: dict[str, Any]) -> tuple[bool, str]:
    module = _load_module(tree, "scripts/hyp_losses.py", "cgv_parity_hyp_losses")
    symbol = component["symbol"]
    ok = hasattr(module, symbol)
    return ok, (f"{symbol} is defined in scripts/hyp_losses.py" if ok
                else f"{symbol} is MISSING from scripts/hyp_losses.py")


def _half_scoped_term(tree: Path, component: dict[str, Any]) -> tuple[bool, str]:
    """The role separation, measured: the term's gradient must leave one half exactly zero."""
    if str(CGV_DIR) not in sys.path:
        sys.path.insert(0, str(CGV_DIR))
    import engine  # the verifier's own loader, so the arm is loaded the way the gates load it

    trainer, saved_path, saved_mods = engine.load_trainer(tree)
    try:
        torch.manual_seed(3)
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=32)
        config = trainer._HYP_ARM_CONFIGS[trainer.EXPERIMENT_ARM]
        count = 16
        latent = torch.randn(2 * count, 32)
        codebook = model.rq.vq_layers[0].get_code_embs().detach().clone().requires_grad_(True)
        positive = torch.arange(count)
        degrees = torch.ones(count)
        term = trainer._hyp_mechanism_loss(component["symbol"], latent, codebook, positive,
                                          config["curvature"], config["strength"], degrees,
                                          euclidean=False)
        term.backward()
        gradient = codebook.grad
        if gradient is None:
            return False, "the term does not reach the level-0 codebook at all"
        half = codebook.shape[1] // 2
        content = float(gradient[:, :half].abs().max())
        behaviour = float(gradient[:, half:].abs().max())
        ok = content == 0.0 and behaviour > 0.0
        return ok, (f"the content half's gradient is exactly {content} and the behaviour half's is "
                    f"{behaviour:.6e}")
    finally:
        engine.unload_trainer(saved_path, saved_mods)


def _run_component(tree: Path, component: dict) -> tuple[bool, str]:
    kind = component["kind"]
    if kind == "symbol_present":
        return _symbol_present(tree, component)
    if kind == "active_mechanism":
        module = _load_module(tree, "scripts/hyp_losses.py", "cgv_parity_active")
        symbol = component["symbol"]
        active = [name for name, entry in module.ARM_CONFIGS.items()
                  if entry.get("mechanism") == symbol]
        return bool(active), (f"the arm's configuration uses the mechanism {symbol!r}" if active
                              else f"the mechanism {symbol!r} exists but is NOT the arm's active "
                                   f"mechanism: the tree would run a reduced mechanism")
    if kind == "half_scoped_term":
        return _half_scoped_term(tree, component)
    return False, f"unknown component kind {kind!r}"


def check(tree: Path | str, mechanism_id: str) -> dict[str, Any]:
    """Whether the tree carries every component the mechanism declares."""
    tree = Path(tree)
    declared = COMPONENTS.get(mechanism_id)
    if declared is None:
        return {"ok": False, "mechanism": mechanism_id, "tree": str(tree), "components": [],
                "missing": ["the mechanism declares no components, so nothing can be checked"],
                "verdict": "no components declared"}
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for component in declared:
        ok, detail = _run_component(tree, component)
        results.append({"id": component["id"], "kind": component["kind"],
                        "symbol": component["symbol"], "ok": bool(ok), "detail": detail,
                        "why": component["why"]})
        if not ok:
            missing.append(component["id"])
    if missing:
        verdict = (f"the implementation is missing {missing}: the proposal declared them and the "
                   f"tree does not carry them, so a join that measured this tree would be measuring "
                   f"a reduced mechanism")
    else:
        verdict = "the implementation carries every component the proposal declared"
    return {"ok": not missing, "mechanism": mechanism_id, "tree": str(tree),
            "components": results, "missing": missing, "verdict": verdict}


__all__ = ["COMPONENTS", "check"]
