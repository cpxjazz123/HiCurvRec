"""Task #180 Stage 1 helper — 把 item_emb_graph.npy 转为 parquet (跟 item_emb.parquet 同 schema).

train_hrqvae.py 期望 parquet 包含 'embedding' 列 (numpy array per row).
我们写一个简单转换脚本:
  - 读 item_emb_graph.npy (9922, 768)
  - 写 item_emb_graph.parquet (9922 行, 每行 'embedding' 列是 768-d numpy array)
"""
import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("task180_npy_to_parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npy", type=str, required=True)
    parser.add_argument("--output_parquet", type=str, required=True)
    args = parser.parse_args()

    log.info(f"Loading {args.input_npy}")
    embeddings = np.load(args.input_npy)
    n, d = embeddings.shape
    log.info(f"  Shape: ({n}, {d})")

    # 跟原 item_emb.parquet schema 对齐: 'embedding' 列每行是 numpy array
    df = pd.DataFrame({'embedding': list(embeddings)})
    df.to_parquet(args.output_parquet)
    log.info(f"Saved {args.output_parquet}")


if __name__ == '__main__':
    main()