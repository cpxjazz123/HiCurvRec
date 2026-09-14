"""Backward-compatible alias for the RecBole-aligned HG-Rec runner.

Historical versions of this file implemented a separate DDP protocol
(different SID offsets, no user token, beam=20, and token-level evaluation).
Keeping that code reachable would make an experiment depend on the entry point
used, so this alias forwards to ``train_HG-Rec.py`` and only translates the old
path arguments.
"""

import argparse
import os
import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="RecBole-aligned HG-Rec DDP alias")
    parser.add_argument(
        "--sid_npy",
        type=str,
        default=str(SCRIPT_DIR / "dataset/Amazon_2023_Instruments/item_sids_recbole.json"),
    )
    parser.add_argument("--product_dir", type=str, default="./ckpt/Amazon_2023_Instruments")
    parser.add_argument("--tag", type=str, default="baseline")
    args, passthrough = parser.parse_known_args()

    os.chdir(SCRIPT_DIR)
    save_path = Path(args.product_dir) / args.tag
    log_path = Path(args.product_dir) / f"{args.tag}_logs"
    translated = [
        str(SCRIPT_DIR / "train_HG-Rec.py"),
        "--code_path",
        args.sid_npy,
        "--save_path",
        str(save_path),
        "--log_path",
        str(log_path),
        "--train_file",
        "train_recbole.parquet",
        "--valid_file",
        "valid_recbole.parquet",
        "--test_file",
        "test_recbole.parquet",
        *passthrough,
    ]
    sys.argv = translated
    runpy.run_path(str(SCRIPT_DIR / "train_HG-Rec.py"), run_name="__main__")


if __name__ == "__main__":
    main()
