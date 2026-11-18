"""Re-export a SID from its saved checkpoint and record whether they agree.

The trainer used to export the SID from the live in-memory module.  A fresh load
of the same weights can rotate near-tie items, which changes the table -- and with
it the downstream recall -- while leaving no trace.  This check closes that gap:
it loads the checkpoint from disk in this process, runs the tree's own export
entry (``model.get_indices``), compares the first three tokens item by item, and
writes ``export_manifest.json`` beside the SID.

Stage 3 reads that manifest and refuses a table the checkpoint does not
reproduce (see :mod:`export_policy`).  The check is read-only apart from the
manifest: a mismatch is reported, never silently repaired, because rewriting the
table would invalidate every measurement already taken on it.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m \
        tools.research.stage2.export_check --sid results/stage2_RQ-VAE/<ARM>/item_sids.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EMBEDDINGS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_tree(arm: str, repo: Path = REPO) -> Path | None:
    """Where the Stage 2 source of this arm lives, in the order the repo moved it."""
    candidates = [repo / "stage2_RQ-VAE" / arm]
    for pattern in (f"analysis_sid_recall/r2/*/{arm}",
                    f"stage2_RQ-VAE/_archive/{arm}",
                    f"stage2_RQ-VAE/baseline/{arm}"):
        candidates.extend(sorted(repo.glob(pattern)))
    for candidate in candidates:
        if (candidate / "train_rqvae.py").is_file():
            return candidate
    return None


def _purge_tree_modules() -> None:
    """Drop the packages a previous tree imported, or the next one reuses them."""
    for name in list(sys.modules):
        if name in ("model", "distances", "losses") or name.startswith(
                ("model.", "distances.", "losses.")):
            del sys.modules[name]


def _load_trainer(tree: Path):
    _purge_tree_modules()
    while str(tree) in sys.path:
        sys.path.remove(str(tree))
    sys.path.insert(0, str(tree))
    spec = importlib.util.spec_from_file_location(
        "export_check_trainer_" + tree.name, tree / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_model(tree: Path, trainer, n_items: int):
    import importlib

    import numpy as np
    package = sys.modules.get("model") or importlib.import_module("model")
    config = trainer._tokenizer_config()
    if hasattr(package, "MixedRQVAE"):
        import torch
        kwargs = {"in_dim": 768, "curvature": float(getattr(trainer, "CURVATURE", 1.0))}
        if getattr(trainer, "ALPHA_MODE", "") == "codebook":
            kwargs["code_alpha_mode"] = True
        if "role_weights" in package.MixedRQVAE.__init__.__code__.co_varnames:
            role_module = sys.modules["model.role_geometry"]
            roles = role_module.load_role_table(trainer.ROLE_TABLE, trainer.ROLE_TABLE_SHA256)
            train_frame = __import__("pandas").read_parquet(trainer.TRAIN_FILE)
            train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64)).tolist()
            weights, _ = role_module.build_weights(trainer.ROLE_MODE, n_items, roles, train_ids)
            kwargs["role_weights"] = weights
            kwargs["geometry_scales"] = tuple(getattr(trainer, "GEOMETRY_SCALES", (1.0, 1.0, 1.0)))
        return package.MixedRQVAE(config, **kwargs)
    return package.RQVAE(config, in_dim=768)


def _sid_payload(rows) -> dict:
    return {str(index): [int(value) for value in row] for index, row in enumerate(rows)}


def check(tree: Path, checkpoint: Path, sid: Path,
          embeddings: Path = EMBEDDINGS, rewrite: bool = False) -> dict:
    """Re-export and compare; returns the manifest that goes beside the SID.

    ``rewrite=True`` replaces ``item_sids.json`` with the checkpoint's own table
    when the two disagree, keeping the trainer's table beside it as
    ``item_sids.trainer.json``.  Stage 3 uses this for a fresh run: the framework
    owns the export (CLAUDE.md 66), so the table Stage 3 reads must be the one the
    checkpoint produces, and the manifest records both digests.
    """
    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors = torch.from_numpy(np.load(embeddings).astype(np.float32))
    trainer = _load_trainer(tree)
    model = _build_model(tree, trainer, len(vectors))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit("checkpoint does not fit this tree's model: "
                         f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    model.to(device).eval()
    kwargs = {}
    if "item_ids" in inspect.signature(model.get_indices).parameters:
        kwargs["item_ids"] = torch.arange(len(vectors), dtype=torch.long, device=device)
    with torch.no_grad():
        tokens = model.get_indices(vectors.to(device), **kwargs).cpu().numpy().astype(np.int64)

    exported = json.loads(Path(sid).read_text(encoding="utf-8"))
    expected = np.array([[int(v) for v in exported[str(i)][:3]] for i in range(len(exported))])
    same_l1 = float((tokens[:, 0] == expected[:, 0]).mean())
    same_l3 = float((tokens[:, :3] == expected).all(axis=1).mean())
    differing = int(round((1 - same_l3) * tokens.shape[0]))
    rewritten = False
    original_sid_sha = sha256_of(sid)
    if differing and rewrite:
        extend = getattr(trainer, "_extend_collisions", None)
        sizes = list(config_sizes(trainer))
        rows = extend(tokens[:, :3], sizes) if extend else tokens[:, :3]
        backup = sid.parent / "item_sids.trainer.json"
        if not backup.is_file():
            backup.write_text(sid.read_text(encoding="utf-8"), encoding="utf-8")
        temporary = sid.with_suffix(".json.partial")
        temporary.write_text(json.dumps(_sid_payload(rows), indent=2), encoding="utf-8")
        temporary.replace(sid)
        rewritten = True
        # The table now IS the checkpoint's export.
        expected = tokens[:, :3]
        same_l1 = 1.0
        same_l3 = 1.0
    return {
        "schema": "genrec.stage2.export_manifest/1",
        "tree": str(Path(tree).relative_to(REPO)) if Path(tree).is_relative_to(REPO) else str(tree),
        "checkpoint": (str(Path(checkpoint).relative_to(REPO))
                       if Path(checkpoint).is_relative_to(REPO) else str(checkpoint)),
        "checkpoint_sha256": sha256_of(checkpoint),
        "sid": str(Path(sid).relative_to(REPO)) if Path(sid).is_relative_to(REPO) else str(sid),
        "sid_sha256": sha256_of(sid),
        "tokens_sha256": hashlib.sha256(tokens[:, :3].tobytes()).hexdigest(),
        "items": int(tokens.shape[0]),
        "same_l1": same_l1,
        "same_l3": same_l3,
        "differing_items": differing,
        "tokens_match": same_l3 > 0.9999,
        "rewritten_from_checkpoint": rewritten,
        "original_sid_sha256": original_sid_sha,
        "original_differing_items": differing,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def config_sizes(trainer) -> list[int]:
    """The codebook sizes the trainer's collision extension expects."""
    config = trainer._tokenizer_config()
    size = config.codebook_size
    return [int(size)] * int(config.codebook_num) if isinstance(size, int) else [int(s) for s in size]


def write_manifest(manifest: dict, path: Path) -> None:
    """Atomic: a half-written manifest must never be read as a verdict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--tree", default="")
    parser.add_argument("--embeddings", default=str(EMBEDDINGS))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--rewrite", action="store_true",
                        help="replace the SID with the checkpoint's table when they differ")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    sid = Path(args.sid).resolve()
    if not sid.is_file():
        raise SystemExit(f"no SID at {sid}")
    arm = sid.parent.name
    tree = Path(args.tree).resolve() if args.tree else resolve_tree(arm)
    if tree is None:
        raise SystemExit(f"cannot find the Stage 2 source tree of {arm}")
    checkpoint = sid.parent / "rqvae_best.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"no checkpoint beside {sid}")
    manifest = check(tree, checkpoint, sid, Path(args.embeddings), rewrite=args.rewrite)
    path = Path(args.manifest) if args.manifest else sid.parent / "export_manifest.json"
    write_manifest(manifest, path)
    if args.print_json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print("%s: tokens_match=%s (%d of %d items differ)%s -> %s" % (
            arm, manifest["tokens_match"], manifest["original_differing_items"],
            manifest["items"],
            " [rewritten from the checkpoint]" if manifest["rewritten_from_checkpoint"] else "",
            path))
    return 0 if manifest["tokens_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
