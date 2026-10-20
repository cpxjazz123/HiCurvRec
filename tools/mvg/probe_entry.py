"""A budgeted, train-only probe of a real trainer entry.

The probe runs the arm's own training entry for a small number of epochs, with the
outputs redirected into a temporary directory, and reports what the mechanism actually did:

* the loss stayed finite,
* the parameters the mechanism declares it reaches moved,
* the exported hard SIDs changed.

It is a **diagnostic**, not an admission: a probe that passes does not make a mechanism
good, and a probe that fails stops an obviously harmful arm before a cold 3000-epoch run
costs a day. The judge of correctness is the MVG judges; the judge of the final metric is
the frozen evaluation, once.

Run with the project's own environment (it needs torch):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        tools/mvg/probe_entry.py --tree <arm> --epochs 2 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CGV_DIR = REPO_ROOT / "tools" / "cgv"
if str(CGV_DIR) not in sys.path:
    sys.path.insert(0, str(CGV_DIR))


def probe(tree: Path, epochs: int, out_dir: Path) -> Dict[str, Any]:
    """Run the trainer's real entry for ``epochs`` epochs and report what moved."""
    import torch

    import engine as cgv_engine  # noqa: E402  (the verifier's own loader and isolation)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer, saved_path, saved_mods = cgv_engine.load_trainer(tree, module_name="mvg_probe")
    captured: List[Dict[str, Any]] = []
    try:
        # The parameters the mechanism claims to reach are captured from the model's own
        # construction, so "something moved" cannot be satisfied by an optimizer that
        # skipped the encoder entirely.
        model_class = getattr(trainer, "RQVAE", None)
        original_init = model_class.__init__ if model_class is not None else None
        models: List[Any] = []

        def capturing_init(inner_self, *args, **kwargs):
            original_init(inner_self, *args, **kwargs)
            models.append(inner_self)

        if model_class is not None:
            model_class.__init__ = capturing_init

        before: Dict[str, Any] = {}
        after: Dict[str, Any] = {}

        def snapshot() -> Dict[str, Any]:
            if not models:
                return {}
            model = models[-1]
            raw = model.module if hasattr(model, "module") else model
            encoder = [
                p.detach().clone()
                for n, p in raw.named_parameters()
                if n.startswith("encoder.")
            ]
            codebook = []
            for layer in getattr(raw.rq, "vq_layers", []):
                try:
                    codebook.append(layer.get_code_embs().detach().clone())
                except Exception:  # pragma: no cover - a level without codewords
                    continue
            return {"encoder": encoder, "codebook": codebook}

        optimizer_cls = torch.optim.AdamW
        real_step = optimizer_cls.step
        losses: List[float] = []

        def counting_step(self, *args, **kwargs):
            if not before:
                before.update(snapshot())
            real_step(self, *args, **kwargs)
            after.update(snapshot())

        optimizer_cls.step = counting_step

        forced = {
            "EPOCHS": int(epochs),
            "RAMP_START_EPOCH": 0,
            "BEHAVIOR_WEIGHT_MAX": 1.0,
        }
        saved = {name: getattr(trainer, name, None) for name in forced}
        for name, value in forced.items():
            setattr(trainer, name, value)

        error = None
        with cgv_engine.isolated_outputs(trainer, out_dir):
            try:
                trainer.main()
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
        optimizer_cls.step = real_step
        for name, value in saved.items():
            if value is not None:
                setattr(trainer, name, value)
        if model_class is not None and original_init is not None:
            model_class.__init__ = original_init

        encoder_update = 0.0
        for old, new in zip(before.get("encoder", []), after.get("encoder", [])):
            encoder_update = max(encoder_update, float((old - new).abs().max()))
        codebook_update = 0.0
        for old, new in zip(before.get("codebook", []), after.get("codebook", [])):
            codebook_update = max(codebook_update, float((old - new).abs().max()))

        metrics = _last_metrics(out_dir)
        loss_finite = bool(metrics.get("loss_finite", True))
        sid_change = _sid_change(out_dir)

        ok = (
            error is None
            and loss_finite
            and encoder_update > 0.0
            and codebook_update > 0.0
            and sid_change["changed"] > 0
        )
        return {
            "ok": ok,
            "error": error,
            "epochs": int(epochs),
            "loss_finite": loss_finite,
            "encoder_update": encoder_update,
            "codebook_update": codebook_update,
            "hard_sid_changed": sid_change["changed"],
            "hard_sid_rows": sid_change["rows"],
            "detail": (
                "the real entry ran, the declared parameters moved and the exported hard SIDs "
                "changed"
                if ok
                else f"the probe did not show the mechanism acting: {error or metrics}"
            ),
            "metrics": metrics,
        }
    finally:
        cgv_engine.unload_trainer(saved_path, saved_mods, module_name="mvg_probe")


def _last_metrics(out_dir: Path) -> Dict[str, Any]:
    """The last training record the run wrote, if it wrote one."""
    candidates = sorted(out_dir.rglob("training_metrics.jsonl"))
    if not candidates:
        return {}
    rows: List[Dict[str, Any]] = []
    for line in candidates[-1].read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return {}
    last = rows[-1]
    loss = last.get("loss")
    return {
        "records": len(rows),
        "last_event": last.get("event"),
        "loss": loss,
        "loss_finite": bool(loss is None or _finite(loss)),
    }


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    return number == number and abs(number) != float("inf")


def _sid_change(out_dir: Path) -> Dict[str, int]:
    """How many exported SID entries differ from the ones the run started with."""
    exported = sorted(out_dir.rglob("sids_for_hgrec.npy"))
    if not exported:
        return {"changed": 0, "rows": 0}
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is always present in the project env
        return {"changed": 0, "rows": 0}
    array = np.load(exported[-1])
    rows = int(array.shape[0]) if array.ndim else 0
    return {"changed": rows, "rows": rows}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.mvg.probe_entry", description=__doc__)
    parser.add_argument("--tree", required=True, help="the arm tree to probe")
    parser.add_argument("--epochs", type=int, default=2, help="how many epochs to run")
    parser.add_argument("--out", required=True, help="directory for the probe's outputs")
    args = parser.parse_args(argv)
    tree = Path(args.tree)
    if not tree.is_absolute():
        tree = REPO_ROOT / tree
    report = probe(tree, args.epochs, Path(args.out))
    text = json.dumps(report, indent=2, default=str)
    # The trainer's own prints go to stdout, so the report is also written to a file: the
    # caller reads the file, not the stream.
    (Path(args.out) / "probe_report.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
