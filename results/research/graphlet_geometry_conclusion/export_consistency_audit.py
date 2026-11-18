"""Is the archived SID what the archived checkpoint actually exports?

Every number in the R2 archive holds a checkpoint and an `item_sids.json`.  The
two are only meaningful together if the table is reproducible from the weights:
a SID that a post-processing step rewrote, or one exported from a different
checkpoint than the one stored beside it, would make the archived recall
unreproducible.

This script reloads a (tree, checkpoint, SID) triple, calls the tree's own export
path -- `model.get_indices`, which is exactly what the trainer used for
`item_sids.json` -- and compares the first three tokens item by item.

Two modes:

    # the arms that still have local artifacts
    python3 .../export_consistency_audit.py

    # a directory of downloaded archives, <root>/<NN>/{mechanism/,item_sids.json,rqvae_best.pth}
    python3 .../export_consistency_audit.py --archive-root results/research/issue139/_r2_download
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
STAGE2 = REPO / "results" / "stage2_RQ-VAE"
R2 = REPO / "analysis_sid_recall" / "r2"


def arm(name, tree, checkpoint, sid):
    return {"name": name, "tree": tree, "checkpoint": checkpoint, "sid": sid}


ARMS = [
    arm("TIGER_RQ-VAE", REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE",
        STAGE2 / "TIGER_RQ-VAE" / "rqvae_best.pth", STAGE2 / "TIGER_RQ-VAE" / "item_sids.json"),
    arm("TIGER_RQ-VAE__canonical", REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE",
        STAGE2 / "TIGER_RQ-VAE__canonical" / "rqvae_best.pth",
        STAGE2 / "TIGER_RQ-VAE__canonical" / "item_sids.json"),
    arm("TIGER_mixed_rq_F0", R2 / "19" / "TIGER_mixed_rq_F0",
        STAGE2 / "TIGER_mixed_rq_F0" / "rqvae_best.pth", STAGE2 / "TIGER_mixed_rq_F0" / "item_sids.json"),
    arm("TIGER_mixed_rq_E", R2 / "16" / "TIGER_mixed_rq_E", R2 / "16" / "rqvae_best.pth",
        R2 / "16" / "item_sids.json"),
    arm("TIGER_mixed_rq_G", R2 / "18" / "TIGER_mixed_rq_G", R2 / "18" / "rqvae_best.pth",
        R2 / "18" / "item_sids.json"),
    arm("TIGER_mixed_rq_L", R2 / "15" / "TIGER_mixed_rq_L", R2 / "15" / "rqvae_best.pth",
        R2 / "15" / "item_sids.json"),
    arm("TIGER_mixed_rq_A", R2 / "17" / "TIGER_mixed_rq_A", R2 / "17" / "rqvae_best.pth",
        R2 / "17" / "item_sids.json"),
    arm("TIGER_role_geom_rq", R2 / "70" / "TIGER_role_geom_rq", R2 / "70" / "rqvae_best.pth",
        STAGE2 / "TIGER_role_geom_rq" / "item_sids.json"),
    arm("TIGER_role_geom_global_rq", R2 / "71" / "TIGER_role_geom_global_rq",
        R2 / "71" / "rqvae_best.pth", STAGE2 / "TIGER_role_geom_global_rq" / "item_sids.json"),
    arm("TIGER_role_geom_shuffled_rq", R2 / "72" / "TIGER_role_geom_shuffled_rq",
        R2 / "72" / "rqvae_best.pth", STAGE2 / "TIGER_role_geom_shuffled_rq" / "item_sids.json"),
    arm("TIGER_graphlet_codebook_rq", REPO / "stage2_RQ-VAE" / "TIGER_graphlet_codebook_rq",
        STAGE2 / "TIGER_graphlet_codebook_rq" / "rqvae_best.pth",
        STAGE2 / "TIGER_graphlet_codebook_rq" / "item_sids.json"),
    arm("TIGER_graphlet_codebook_global_rq",
        REPO / "stage2_RQ-VAE" / "TIGER_graphlet_codebook_global_rq",
        STAGE2 / "TIGER_graphlet_codebook_global_rq" / "rqvae_best.pth",
        STAGE2 / "TIGER_graphlet_codebook_global_rq" / "item_sids.json"),
    arm("TIGER_Uniform_KNN_RQ", REPO / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ",
        STAGE2 / "TIGER_Uniform_KNN_RQ" / "rqvae_best.pth",
        STAGE2 / "TIGER_Uniform_KNN_RQ" / "item_sids.json"),
    arm("TIGER_Uniform_KNN_control", REPO / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_control",
        STAGE2 / "TIGER_Uniform_KNN_control" / "rqvae_best.pth",
        STAGE2 / "TIGER_Uniform_KNN_control" / "item_sids.json"),
]


def archive_specs(root: Path) -> list[dict]:
    """Downloaded archives: <root>/<NN>/{<mechanism>/train_rqvae.py, item_sids.json, rqvae_best.pth}."""
    specs = []
    for number_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sid = number_dir / "item_sids.json"
        checkpoint = number_dir / "rqvae_best.pth"
        trees = sorted(p for p in number_dir.iterdir()
                       if p.is_dir() and (p / "train_rqvae.py").is_file())
        if not sid.is_file() or not checkpoint.is_file() or not trees:
            specs.append({"name": number_dir.name, "tree": None,
                          "checkpoint": checkpoint, "sid": sid,
                          "error": "missing sid/checkpoint/source"})
            continue
        specs.append(arm(f"{number_dir.name}:{trees[0].name}", trees[0], checkpoint, sid))
    return specs


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def purge_tree_modules() -> None:
    """Drop the packages a previous tree imported, or the next one reuses them.

    Every trainer imports `model` (and the baseline also `distances`/`losses`) as
    top-level packages rooted at its own tree.  Leaving them in `sys.modules`
    makes the second arm audit the first arm's classes, which is exactly the bug
    this guard exists for.
    """
    for name in list(sys.modules):
        if name in ("model", "distances", "losses") or name.startswith(
                ("model.", "distances.", "losses.")):
            del sys.modules[name]


def load_trainer(tree: Path):
    purge_tree_modules()
    while str(tree) in sys.path:
        sys.path.remove(str(tree))
    sys.path.insert(0, str(tree))
    spec = importlib.util.spec_from_file_location(
        "audit_trainer_" + tree.name, tree / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(tree: Path, trainer, n_items: int, device):
    """The model class the trainer itself imported from this tree."""
    package = sys.modules["model"]
    config = trainer._tokenizer_config()
    if hasattr(package, "MixedRQVAE"):
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


def audit_one(spec, embeddings, device) -> dict:
    if spec.get("tree") is None:
        return {"verdict": "AUDIT_FAILED", "error": spec.get("error", "no source tree")}
    if not spec["checkpoint"].is_file() or not spec["sid"].is_file():
        return {"verdict": "AUDIT_FAILED", "error": "missing checkpoint or sid"}
    trainer = load_trainer(spec["tree"])
    model = build_model(spec["tree"], trainer, len(embeddings), device)
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        return {"verdict": "AUDIT_FAILED", "error": "state mismatch",
                "missing": list(missing)[:5], "unexpected": list(unexpected)[:5]}
    model.to(device).eval()
    kwargs = {}
    if "item_ids" in inspect.signature(model.get_indices).parameters:
        kwargs["item_ids"] = torch.arange(len(embeddings), dtype=torch.long, device=device)
    with torch.no_grad():
        tokens = model.get_indices(embeddings.to(device), **kwargs).cpu().numpy().astype(np.int64)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    exported = json.loads(spec["sid"].read_text(encoding="utf-8"))
    expected = np.array([[int(v) for v in exported[str(i)][:3]] for i in range(len(exported))])
    same_l1 = float((tokens[:, 0] == expected[:, 0]).mean())
    same_l3 = float((tokens[:, :3] == expected).all(axis=1).mean())
    return {
        "items": int(tokens.shape[0]), "same_l1": same_l1, "same_l3": same_l3,
        "differing_items": int(round((1 - same_l3) * tokens.shape[0])),
        "checkpoint_sha256": sha256_of(spec["checkpoint"]),
        "sid_sha256": sha256_of(spec["sid"]),
        "verdict": "REPRODUCIBLE" if same_l3 > 0.9999 else "NOT_REPRODUCIBLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.archive_root:
        specs = archive_specs(Path(args.archive_root))
        print(f"{len(specs)} archive numbers under {args.archive_root}")
    else:
        specs = ARMS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = torch.from_numpy(np.load(EMBEDDING).astype(np.float32))
    report = {"schema": "genrec.export_consistency_audit.v1", "mode": args.archive_root or "local",
              "arms": {}}
    print(f"{'number/arm':52s} {'same L1':>8s} {'same L3':>8s} {'diff':>5s}  verdict")
    for spec in specs:
        try:
            entry = audit_one(spec, embeddings, device)
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            entry = {"verdict": "AUDIT_FAILED", "error": repr(exc)}
        report["arms"][spec["name"]] = entry
        print(f"{spec['name']:52s} {entry.get('same_l1', 0):8.4f} {entry.get('same_l3', 0):8.4f} "
              f"{entry.get('differing_items', '-'):>5}  {entry['verdict']}"
              + (f"  {entry['error'][:60]}" if entry.get("error") else ""), flush=True)

    out = Path(args.out) if args.out else (HERE / ("export_consistency_audit.json" if not args.archive_root
                                                   else "export_consistency_archive.json"))
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    verdicts = [entry["verdict"] for entry in report["arms"].values()]
    print(f"\nreproducible {verdicts.count('REPRODUCIBLE')} / {len(verdicts)}; "
          f"failed {verdicts.count('AUDIT_FAILED')}; "
          f"not reproducible {verdicts.count('NOT_REPRODUCIBLE')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
