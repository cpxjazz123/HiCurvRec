"""Issue #266: cross-bind R2 108/109, local source/result, token cache and graph."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEDGER = ROOT / "results/research/issue139/r2_ledger.json"
ARMS = {
    108: "TIGER_frozen_topology_gnn_rq",
    109: "TIGER_euclid_local_curve_supervision_rq",
}
RUNS = {
    108: "stage2_ae668bc04dc44d52a8bed5f9408e8e74",
    109: "stage2_4891fa8df3e843839de770e52e156892",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def one_final(arm: str) -> Path:
    paths = sorted((ROOT / "results/stage3_T5Train" / arm).glob("**/test_final.json"))
    if len(paths) != 1:
        raise RuntimeError(f"{arm}: expected one test_final.json, got {len(paths)}")
    return paths[0]


def check_file(label: str, actual: Path, expected: str, checks: list) -> None:
    ok = actual.is_file() and sha(actual) == expected
    checks.append({"label": label, "path": str(actual), "expected_sha256": expected,
                   "actual_sha256": sha(actual) if actual.is_file() else None,
                   "status": "PASS" if ok else "FAIL"})


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    result = {"schema": "genrec.issue266.identity/1", "issue": 266,
              "mode": "read-only", "arms": {}}
    for index, arm in ARMS.items():
        entry = ledger[str(index)]
        bundle = HERE / "_r2" / str(index)
        run = ROOT / "results/stage2_RQ-VAE/_runs" / RUNS[index] / arm
        source = run / "src"
        checks = []
        # 1) All R2 objects listed in the frozen ledger.
        for relative, expected in entry["files"].items():
            check_file(f"r2:{relative}", bundle / relative, expected, checks)
        # 2) R2 top-level four pieces.
        for name in ("item_sids.json", "recall.json", "rqvae_best.pth"):
            check_file(f"r2:{name}", bundle / name, entry["files"][name], checks)
        # 3) Local source and result against the same R2 hashes.
        result_paths = {
            "item_sids.json": run / "result/item_sids.json",
            "rqvae_best.pth": run / "result/rqvae_best.pth",
            "recall.json": bundle / "recall.json",
        }
        r2_only = []
        for relative, expected in entry["files"].items():
            if not relative.startswith(arm + "/"):
                continue
            relative = relative[len(arm) + 1:]
            if relative.startswith("assets/"):
                target = ROOT / "assets" / relative[len("assets/"):]
            elif relative in result_paths:
                target = result_paths[relative]
            elif relative.startswith(("results/", "tools/", "stage2_source.tar.gz")):
                r2_only.append(relative)
                continue
            else:
                target = source / relative
            if not target.is_file() and relative not in result_paths and not relative.startswith("assets/"):
                r2_only.append(relative)
                continue
            check_file(f"local:{relative}", target, expected, checks)
        # 4) test_final identity and archived recall.
        final = one_final(arm)
        fd = json.loads(final.read_text())
        recall = json.loads((bundle / "recall.json").read_text())
        bound_checkpoint = Path(fd["stage2_checkpoint"])
        bound_sid = Path(fd["sid_path"])
        recall_identity = {
            "final": str(final), "final_sha256": sha(final),
            "test_recall@10": float(fd["test_recall@10"]),
            "test_recall@5": float(fd["test_recall@5"]),
            "test_ndcg@10": float(fd["test_ndcg@10"]),
            "n_eval": int(recall["n_eval"]),
            "checkpoint_path_match": bound_checkpoint == result_paths["rqvae_best.pth"],
            "sid_path_match": bound_sid == result_paths["item_sids.json"],
            "recall_matches_final": (
                float(recall["test_recall@10"]) == float(fd["test_recall@10"])
                and float(recall["test_recall@5"]) == float(fd["test_recall@5"])
                and float(recall["test_ndcg@10"]) == float(fd["test_ndcg@10"])),
            "recall_checkpoint_sha": recall["stage2_checkpoint_sha256"],
            "bound_checkpoint_sha": sha(bound_checkpoint),
            "recall_sid_sha": recall["sid_sha256"],
            "bound_sid_sha": sha(bound_sid),
        }
        if not (recall_identity["checkpoint_path_match"] and recall_identity["sid_path_match"]
                and recall_identity["recall_matches_final"]
                and recall_identity["recall_checkpoint_sha"] == recall_identity["bound_checkpoint_sha"]
                and recall_identity["recall_sid_sha"] == recall_identity["bound_sid_sha"]):
            raise RuntimeError(f"{arm}: downstream identity mismatch")
        # 5) Token cache and its graph/asset identity.
        token_manifest_path = ROOT / "results/stage2_RQ-VAE/TIGER_frozen_topology_gnn_rq/token_manifest.json"
        token_manifest = json.loads(token_manifest_path.read_text())
        token_cache = Path(token_manifest["cache"]["path"])
        graph = Path(token_manifest["identity"]["graph"])
        curvature = Path(token_manifest["identity"]["curvature"])
        token_checks = {
            "token_manifest": str(token_manifest_path), "token_manifest_sha256": sha(token_manifest_path),
            "token_cache": str(token_cache), "token_cache_sha256": sha(token_cache),
            "token_cache_expected": token_manifest["cache"]["sha256"],
            "token_cache_match": sha(token_cache) == token_manifest["cache"]["sha256"],
            "tokens_shape": token_manifest["cache"]["tokens_shape"],
            "mask_shape": token_manifest["cache"]["mask_shape"],
            "valid_tokens": token_manifest["cache"]["valid_tokens"],
            "graph": str(graph), "graph_sha256": sha(graph),
            "graph_expected": token_manifest["identity"]["graph_sha256"],
            "graph_match": sha(graph) == token_manifest["identity"]["graph_sha256"],
            "curvature": str(curvature), "curvature_sha256": sha(curvature),
            "curvature_expected": token_manifest["identity"]["curvature_sha256"],
            "curvature_match": sha(curvature) == token_manifest["identity"]["curvature_sha256"],
        }
        if not (token_checks["token_cache_match"] and token_checks["graph_match"]
                and token_checks["curvature_match"]):
            raise RuntimeError(f"{arm}: token/graph asset mismatch")
        failures = [item for item in checks if item["status"] != "PASS"]
        result["arms"][str(index)] = {
            "arm": arm,
            "r2_root": str(bundle),
            "ledger_files": len(entry["files"]),
            "checks": checks,
            "r2_only": r2_only,
            "r2_only_count": len(r2_only),
            "failures": failures,
            "downstream": recall_identity,
            "tokens": token_checks,
            "status": "PASS" if not failures else "FAIL",
        }
        print(json.dumps({"index": index, "checks": len(checks),
                          "failures": len(failures), "token": token_checks["token_cache_match"],
                          "graph": token_checks["graph_match"],
                          "recall": recall_identity["recall_matches_final"]}, indent=2))
    (HERE / "IDENTITY.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
