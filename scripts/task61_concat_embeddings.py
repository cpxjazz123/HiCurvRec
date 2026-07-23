#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task61_concat_embeddings.py — 拼接 flan-t5 2048d + sentence-t5 768d → 2816d

执行:
  python3 scripts/task61_concat_embeddings.py \
      --input_pt_2048 logs/task59_s1/runs/2026-07-20/04-05-08/pickle/merged_predictions_tensor.pt \
      --input_pt_768 logs/task60_s1/runs/2026-07-20/05-53-06/pickle/merged_predictions_tensor.pt \
      --output_pt logs/task61_s1/merged_predictions_2816d.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt_2048", required=True)
    parser.add_argument("--input_pt_768", required=True)
    parser.add_argument("--output_pt", required=True)
    args = parser.parse_args()

    out = Path(args.output_pt)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[task61 concat] loading {args.input_pt_2048}")
    e1 = torch.load(args.input_pt_2048, map_location="cpu", weights_only=False).float()
    print(f"  shape: {tuple(e1.shape)} dtype: {e1.dtype}")
    print(f"  norm mean: {e1.norm(dim=1).mean():.4f}")

    print(f"[task61 concat] loading {args.input_pt_768}")
    e2 = torch.load(args.input_pt_768, map_location="cpu", weights_only=False).float()
    print(f"  shape: {tuple(e2.shape)} dtype: {e2.dtype}")
    print(f"  norm mean: {e2.norm(dim=1).mean():.4f}")

    if e1.shape[0] != e2.shape[0]:
        raise ValueError(f"item count mismatch: {e1.shape[0]} vs {e2.shape[0]}")

    # mean-center each, then concat
    e1_centered = e1 - e1.mean(dim=0, keepdim=True)
    e2_centered = e2 - e2.mean(dim=0, keepdim=True)
    e_concat = torch.cat([e1_centered, e2_centered], dim=1)  # (n, 2816)

    print(f"[task61 concat] concat shape: {tuple(e_concat.shape)} dtype: {e_concat.dtype}")
    print(f"  norm mean: {e_concat.norm(dim=1).mean():.4f}, std: {e_concat.norm(dim=1).std():.4f}")

    torch.save(e_concat, out)
    print(f"[task61 concat] saved → {out}")


if __name__ == "__main__":
    main()