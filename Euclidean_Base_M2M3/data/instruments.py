"""Musical_Instruments dataset adapter for RQ-VAE-Recommender.

仿照 data/amazon.py 的 AmazonReviews 实现, 但:
- 不下载 (直接读 /home/wlia0047/ar57/wenyu/GeneRec/dataset/ 下的 parquet)
- item embedding 复用已训好的 sentence-t5-xxl 输出 (item_emb.npy)
- 用户序列从 train/valid/test.parquet 转 HeteroData

R44/R47: dataset 路径硬约束, sentence-t5-xxl 在 hj82_scratch2.
"""
import json
import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch_geometric.data import HeteroData, InMemoryDataset


# === 硬编码路径 (R44 dataset 路径约束) ===
DATASET_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/dataset"
ITEM_EMB_NPY = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/item_emb.npy"
ITEM_IDS_JSON = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/item_ids.json"

# Amazon 一样的 max_seq_len (因为 seq_data 内部会切前 max_seq_len 个)
MAX_SEQ_LEN = 20


class RawMusicalInstruments(InMemoryDataset):
    """Musical_Instruments dataset — 9922 items, 10000 users.

    Raw 输入 (来自 GeneRec/dataset/):
      - Instruments.item.json: {item_id_str: {title, description, ...}}
      - train.parquet / valid.parquet / test.parquet: cols=[user, history, target]
        (leave-one-out 三段切分, history 累积)

    Processed 输出 (HeteroData):
      data["item"].x: (9922, 768) sentence-t5-xxl embedding
      data["item"].text: (9922,) 文本描述
      data["item"].is_train: (9922,) 全 True (instruments 都参与 embedding 学习)
      data[("user", "rated", "item")].history: {train/eval/test} 三个 split
        - train.itemId: list[list[int]] 各用户历史 items (可变长度, 训时随机切片)
        - train.itemId_fut: list[int] 下一 item (倒数第二)
        - eval.itemId: padded (max_seq_len,) 用于 eval
        - eval.itemId_fut: list[int] 倒数第二
        - test.itemId: padded (max_seq_len,) 用于 test
        - test.itemId_fut: list[int] 最后一个
    """

    def __init__(
        self,
        root: str,
        split: str = "instruments",  # 占位用, 实际不用 split 区分 (只有一个 split)
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.split = split
        # InMemoryDataset 需要 processed_paths, 我们设到 root/processed/data_instruments.pt
        super().__init__(root, transform, pre_transform, force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_dir(self) -> str:
        # override: 不用 self.root/raw, 直接用 GeneRec/dataset
        return DATASET_ROOT

    @property
    def processed_dir(self) -> str:
        # override: 放到 /home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/processed/
        # 因为 /home 太小 (20G) 不能写大量数据
        return "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/processed"

    @property
    def raw_file_names(self) -> List[str]:
        return ["Instruments.item.json", "train.parquet", "valid.parquet", "test.parquet"]

    @property
    def processed_file_names(self) -> str:
        return f"data_{self.split}.pt"

    def download(self) -> None:
        # No-op: 数据已就绪 (R44 约束)
        pass

    def _build_sequences(self, max_seq_len: int = MAX_SEQ_LEN):
        """从 train/valid/test.parquet 构造序列 split.

        Amazon 格式:
          train.itemId = history[:-2]  (变长 list)
          train.itemId_fut = history[-2]
          eval.itemId = history[-(max_seq_len+2):-2]  (padded 到 max_seq_len, -1 padding)
          eval.itemId_fut = history[-2]
          test.itemId = history[-(max_seq_len+1):-1]  (padded)
          test.itemId_fut = history[-1]

        Instruments 格式 (leave-one-out):
          user 在三个 parquet 都有同一行 (history 累积, target 是 next item).
          设 raw_seq = (train.history + [train.target, valid.target, test.target])
          这样 raw_seq[-3] = train.target, raw_seq[-2] = valid.target, raw_seq[-1] = test.target.

          那么:
            train_seq = raw_seq[:-2]  (含 train.target 不含 valid.target)
            valid_seq = raw_seq[:-1]  (含 valid.target 不含 test.target)
            test_seq  = raw_seq

          itemId_fut:
            train_fut = raw_seq[-2] = valid.target
            eval_fut  = raw_seq[-2] = valid.target  (Amazon 命名: eval 和 train 共享 fut)
            test_fut  = raw_seq[-1] = test.target
        """
        train_df = pd.read_parquet(os.path.join(self.raw_dir, "train.parquet"))
        valid_df = pd.read_parquet(os.path.join(self.raw_dir, "valid.parquet"))
        test_df = pd.read_parquet(os.path.join(self.raw_dir, "test.parquet"))

        assert (train_df["user"].values == valid_df["user"].values).all(), "user alignment broken"
        assert (train_df["user"].values == test_df["user"].values).all(), "user alignment broken"

        n_users = len(train_df)
        # 用 "eval" 键 (Amazon / PreprocessingMixin 约定), 语义上对应 valid (HG-Rec 用法)
        sequences = {sp: {"userId": [], "itemId": [], "itemId_fut": []} for sp in ["train", "eval", "test"]}

        for i in range(n_users):
            user_id = int(train_df.iloc[i]["user"])
            history = list(map(int, train_df.iloc[i]["history"]))
            t_train = int(train_df.iloc[i]["target"])
            t_valid = int(valid_df.iloc[i]["target"])
            t_test  = int(test_df.iloc[i]["target"])

            # item id 从 1 开始 (parquet 1-indexed), 但 codebook/embedding 是 0-indexed (0..9921)
            # 这里全部 -1 转 0-indexed, padding 保持 -1
            history = [h - 1 if h > 0 else -1 for h in history]
            t_train -= 1
            t_valid -= 1
            t_test -= 1

            # 构造完整 raw_seq (历史 + train/valid/test target)
            raw_seq = history + [t_train, t_valid, t_test]

            # train (HG-Rec 公平划分): 输入 = history (不含 target), 标签 = t_train (train.target)
            # 注意: 不能用 raw_seq[:-2] (= history+[t_train]) + 标签 t_valid — 那会让训练标签 = valid 评估标签 (背题)
            train_items = raw_seq[:-3]
            sequences["train"]["userId"].append(user_id)
            sequences["train"]["itemId"].append(train_items)
            sequences["train"]["itemId_fut"].append(t_train)

            # eval (语义上 = HG-Rec valid 集): history[-(max_seq_len+2):-2] (padded -1), fut = t_valid
            eval_items = raw_seq[-(max_seq_len + 2):-2]
            if len(eval_items) < max_seq_len:
                eval_items = [-1] * (max_seq_len - len(eval_items)) + eval_items
            sequences["eval"]["userId"].append(user_id)
            sequences["eval"]["itemId"].append(eval_items)
            sequences["eval"]["itemId_fut"].append(t_valid)

            # test: history[-(max_seq_len+1):-1] (padded -1)
            test_items = raw_seq[-(max_seq_len + 1):-1]
            if len(test_items) < max_seq_len:
                test_items = [-1] * (max_seq_len - len(test_items)) + test_items
            sequences["test"]["userId"].append(user_id)
            sequences["test"]["itemId"].append(test_items)
            sequences["test"]["itemId_fut"].append(t_test)

        # 转 polars DataFrame (与 Amazon 风格一致)
        for sp in sequences:
            sequences[sp] = pl.from_dict(sequences[sp])

        return sequences

    def process(self, max_seq_len: int = MAX_SEQ_LEN) -> None:
        from data.preprocessing import PreprocessingMixin
        data = HeteroData()

        # --- 1. user sequences ---
        sequences = self._build_sequences(max_seq_len=max_seq_len)
        history_dict = {}
        for sp, df in sequences.items():
            # 走 PreprocessingMixin._df_to_tensor_dict 与 Amazon 完全一致:
            #   变长 itemId 保持 list[list[int]], 定长 itemId_fut/userId 转 tensor
            tensor_dict = PreprocessingMixin._df_to_tensor_dict(df, ["itemId"])
            history_dict[sp] = tensor_dict
        data["user", "rated", "item"].history = history_dict

        # --- 2. item features ---
        # 2a. embedding: 直接加载 sentence-t5-xxl 预计算结果 (R47)
        item_emb = torch.from_numpy(np.load(ITEM_EMB_NPY).astype(np.float32))
        assert item_emb.shape == (9922, 768), f"unexpected item_emb shape: {item_emb.shape}"

        # 2b. text: 从 Instruments.item.json 提取
        with open(os.path.join(self.raw_dir, "Instruments.item.json"), "r") as f:
            items_raw = json.load(f)
        item_ids_sorted = sorted(items_raw.keys(), key=lambda x: int(x))
        # 与 item_emb.npy 顺序对齐 (item_ids.json 是排序后的)
        with open(ITEM_IDS_JSON, "r") as f:
            item_ids_emb_order = json.load(f)
        assert item_ids_sorted == item_ids_emb_order, "item_ids order mismatch"

        texts = []
        for iid in item_ids_sorted:
            it = items_raw[iid]
            title = (it.get("title") or "").strip()
            desc = (it.get("description") or "").strip()[:512]
            text = f"{title}. {desc}".strip(". ")
            if not text:
                text = "(no description)"
            texts.append(text)

        data["item"].x = item_emb
        data["item"].text = np.array(texts)
        data["item"].is_train = torch.ones(item_emb.shape[0], dtype=torch.bool)

        # 存 processed
        os.makedirs(self.processed_dir, exist_ok=True)
        self.save([data], self.processed_paths[0])
        print(f"[instruments] processed → {self.processed_paths[0]}", flush=True)
        print(f"  users={len(sequences['train']['userId'])}, items={item_emb.shape[0]}", flush=True)
        for sp in ["train", "eval", "test"]:
            n = len(sequences[sp]["userId"])
            print(f"  split={sp}: {n} users", flush=True)
