"""Prove the floor on the *other* quantiser family: the frozen baseline itself.

Every new Stage2 candidate is copied from `stage2_RQ-VAE/baseline/TIGER_RQ-VAE`, which is a
Euclidean 32-dim VQ with Sinkhorn only on the last level -- a different mechanism family from
the mixed E16xH16 tree the earlier arms used. This script copies that baseline, points it at a
diagnostic result directory, shortens the budget to 300 epochs, records the per-epoch
occupancy of every level, and installs the anticollapse floor.

If the floor is mechanism-agnostic, the baseline's own quantiser must also hold every codeword
at every epoch.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEFAULT_BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
ARM = "B0_floor_euclid"
EPOCHS = 100   # the collapse window is over by epoch 50; 100 is enough to show the guarantee

sys.path.insert(0, str(REPO / "tools" / "research" / "stage2"))
from anticollapse import apply_floor, verify_tree   # noqa: E402

OCCUPANCY_ANCHOR = """        if epoch % EVAL_INTERVAL != 0 and epoch != EPOCHS:"""
OCCUPANCY = """        with torch.no_grad():
            _tokens = raw_module.get_indices(all_embeddings.to(device))
            _used = [int(torch.bincount(_tokens[:, level],
                                        minlength=int(CODEBOOK_SIZE[level])).gt(0).sum())
                     for level in range(len(CODEBOOK_SIZE))]
        _record(rank, "occupancy", epoch=epoch, used=_used)
"""


def main() -> None:
    tree = HERE / "trees" / ARM
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(DEFAULT_BASELINE, tree,
                    ignore=shutil.ignore_patterns("__pycache__", "logs", "observations"),
                    symlinks=True, ignore_dangling_symlinks=True)

    trainer = tree / "train_rqvae.py"
    text = trainer.read_text(encoding="utf-8")
    edits = [
        ('ARM_NAME = "TIGER_RQ-VAE"', f'ARM_NAME = "{ARM}"'),
        ('RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME',
         'RESULT_DIR = REPO_ROOT / "results" / "research" / "qda" / "round9" / "runs" / ARM_NAME'),
        ('LOG_DIR        = REPO_ROOT / "stage2_RQ-VAE" / ARM_NAME / "logs"',
         'LOG_DIR        = REPO_ROOT / "results" / "research" / "qda" / "round9" / "runs" / ARM_NAME / "logs"'),
        # `results/` also contains stage2_RQ-VAE, so one marker stops a level too high;
        # the root needs both, exactly as round 1 found for the mixed tree.
        ("""        if (parent / "stage2_RQ-VAE").is_dir():
            return parent""",
         """        if ((parent / "stage2_RQ-VAE").is_dir()
                and (parent / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy").is_file()):
            return parent"""),
        (OCCUPANCY_ANCHOR, OCCUPANCY + OCCUPANCY_ANCHOR),
    ]
    for old, new in edits:
        assert text.count(old) == 1, f"anchor appears {text.count(old)} times: {old[:80]}"
        text = text.replace(old, new)
    trainer.write_text(text, encoding="utf-8")

    config_path = tree / "config" / "train_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["EPOCHS"] = EPOCHS
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    patch = apply_floor(tree)
    print(json.dumps(patch, indent=2, sort_keys=True))
    print(json.dumps(verify_tree(tree), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
