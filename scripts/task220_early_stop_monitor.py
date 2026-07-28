#!/usr/bin/env python3
"""
Task #220+#221 早期中止 monitor.
监听训练 log, 在 epoch 20 / 50 检查 L0 utilization + collision_rate.
不达标 → kill PID + 写 minimal verdict stub.
"""
import argparse
import os
import re
import sys
import time
import signal
import subprocess
import json
from pathlib import Path

import numpy as np

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_latest_ckpt(ckpt_dir: Path) -> Path | None:
    """找 ckpt_dir 下最新的 epoch_X_model.pth 或 best_loss_model.pth."""
    if not ckpt_dir.exists():
        return None
    pths = list(ckpt_dir.rglob("*.pth"))
    if not pths:
        return None
    # 优先 best_loss_model.pth
    bests = [p for p in pths if "best_loss" in p.name]
    if bests:
        return max(bests, key=lambda p: p.stat().st_mtime)
    return max(pths, key=lambda p: p.stat().st_mtime)


def parse_current_epoch(log_file: Path) -> int:
    """从 log 提取最新已完成 epoch."""
    if not log_file.exists():
        return -1
    text = log_file.read_text(errors="ignore")
    matches = re.findall(r"epoch (\d+) evaluating", text)
    if not matches:
        return -1
    return max(int(m) for m in matches)


def compute_util_and_collision(ckpt_path: Path, assignment_mode: str, c_k_min: float, c_k_max: float) -> dict:
    """加载 ckpt, encode 9922 items, 算 L0 utilization + collision rate."""
    import torch
    import pandas as pd
    import sys
    sys.path.insert(0, str(REPO / "HG-Rec"))
    from model.hrqvae import HRQVAE

    print(f"[monitor] Loading ckpt: {ckpt_path}", flush=True)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # trainer 包装: {args, epoch, best_loss, best_collision_rate, state_dict, optimizer}
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
        print(f"[monitor] unwrap trainer ckpt: epoch={raw.get('epoch')}, best_loss={raw.get('best_loss')}", flush=True)
    else:
        state = raw

    # 自动推断 num_emb_list 从 state_dict keys
    codebook_keys = [k for k in state.keys() if k.startswith("quantizer.vq_layers.")]
    if not codebook_keys:
        codebook_keys = [k for k in state.keys() if "embedding.weight" in k and "vq" in k.lower()]
    layer_sizes = []
    for k in codebook_keys:
        m = re.search(r"vq_layers\.(\d+)\.", k)
        if m:
            layer_sizes.append((int(m.group(1)), k))
    if not layer_sizes:
        # fallback: 从 ckpt_dir 父目录的命名读
        # task220/hrqvae_pck/Jul-26-..._codebook_[64,128,256]_.../best.pth
        m = re.search(r"codebook_\[([\d,]+)\]", str(ckpt_path))
        if m:
            num_emb_list = [int(x) for x in m.group(1).split(",")]
        else:
            raise RuntimeError(f"Cannot infer num_emb_list from {ckpt_path}")
    else:
        # 从 state_dict 读 shapes
        idx_to_size = {}
        for idx, k in layer_sizes:
            shape = state[k].shape
            if len(shape) == 2:
                idx_to_size[idx] = shape[0]
        num_emb_list = [idx_to_size[i] for i in sorted(idx_to_size.keys())]
        # num_emb_list 通常是 [K0, K1, K2]; 如果有 4 个, 取前 3
        if len(num_emb_list) == 4:
            num_emb_list = num_emb_list[:3]

    e_dim = 36
    layers = [512, 256, 128, 36]
    in_dim = 768

    print(f"[monitor] num_emb_list={num_emb_list}, e_dim={e_dim}", flush=True)

    model = HRQVAE(
        in_dim=in_dim,
        num_emb_list=num_emb_list,
        e_dim=e_dim,
        layers=layers,
        loss_type="poincare",
        quant_loss_weight=1.0,
        beta=0.5,
        kmeans_init=False,
        kmeans_iters=10,
        sk_eps=[0.0] * len(num_emb_list),
        sk_iters=50,
        assignment_mode=assignment_mode,
        c_k_min=c_k_min,
        c_k_max=c_k_max,
        c_k_seed=42,
    )
    model.load_state_dict(state, strict=False)
    model.eval()

    # 加载 9922 items
    df = pd.read_parquet(REPO / "HG-Rec/dataset/Instruments/item_emb.parquet")
    if "embedding" in df.columns:
        # 实际格式: 'embedding' 列存 list[float]
        items = torch.tensor(np.stack(df["embedding"].values), dtype=torch.float32)
    else:
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        items = torch.tensor(df[emb_cols].values, dtype=torch.float32)
    print(f"[monitor] Encoding {items.shape[0]} items (dim={items.shape[1]})...", flush=True)

    with torch.no_grad():
        out, rq_loss, indices, path_loss, div_ent = model(items, use_sk=False)

    # indices 是 Tensor (N, num_layers), 每列是 per-layer assignment
    if isinstance(indices, torch.Tensor):
        idx = indices.cpu().numpy()
        assert idx.ndim == 2, f"unexpected indices shape {indices.shape}"
        L0_indices = idx[:, 0]
        L1_indices = idx[:, 1] if idx.shape[1] > 1 else idx[:, 0]
        L2_indices = idx[:, 2] if idx.shape[1] > 2 else idx[:, 0]
    else:
        # 兼容 list 形式
        L0_indices = indices[0].cpu().numpy()
        L1_indices = indices[1].cpu().numpy() if len(indices) > 1 else L0_indices
        L2_indices = indices[2].cpu().numpy() if len(indices) > 2 else L0_indices

    L0_util = len(set(L0_indices.tolist())) / num_emb_list[0]
    L1_util = len(set(L1_indices.tolist())) / num_emb_list[1]
    L2_util = len(set(L2_indices.tolist())) / num_emb_list[2]

    # collision rate: 整个 SID tuple (L0,L1,L2) 重复比例
    sids = list(zip(L0_indices.tolist(), L1_indices.tolist(), L2_indices.tolist()))
    collision_rate = 1 - len(set(sids)) / len(sids)

    return {
        "L0_utilization": L0_util,
        "L1_utilization": L1_util,
        "L2_utilization": L2_util,
        "collision_rate": collision_rate,
        "num_unique_sids": len(set(sids)),
        "num_total": len(sids),
        "ckpt_path": str(ckpt_path),
    }


def write_verdict_stub(task_id: int, verdict_path: Path, reason: str, metrics: dict, epoch: int):
    verdict_path.write_text(
        f"""# Task #{task_id} 早期中止 verdict stub

## 终止原因
{reason}

## 检查 epoch
{epoch}

## 关键指标
```json
{json.dumps(metrics, indent=2)}
```

## 后续动作
1. 检查 products/task{task_id}/hrqvae*/ 下 epoch_{epoch}*.pth 是否存在 (R12 强制存)
2. 决定是否继续训练 / pivot / 写最终 verdict
"""
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--log_file", type=Path, required=True)
    p.add_argument("--pid_file", type=Path, required=True)
    p.add_argument("--ckpt_dir", type=Path, required=True)
    p.add_argument("--check_epochs", type=int, nargs="+", default=[20, 50])
    p.add_argument("--util_kill_below", type=float, nargs="+", default=[0.30, 0.50])
    p.add_argument("--collision_kill_above", type=float, nargs="+", default=[0.95, 0.80])
    p.add_argument("--assignment_mode", type=str, default="shared")
    p.add_argument("--c_k_min", type=float, default=0.5)
    p.add_argument("--c_k_max", type=float, default=5.0)
    p.add_argument("--poll_interval", type=int, default=60)
    args = p.parse_args()

    assert len(args.check_epochs) == len(args.util_kill_below) == len(args.collision_kill_above), \
        "check_epochs/util_kill_below/collision_kill_above 长度必须一致"

    print(f"[monitor-{args.task_id}] started, log={args.log_file}, pid={args.pid_file}, ckpt_dir={args.ckpt_dir}", flush=True)
    print(f"[monitor-{args.task_id}] check_epochs={args.check_epochs}", flush=True)

    fired = set()  # 已触发的 check epoch

    while True:
        # 读 PID
        pid = None
        if args.pid_file.exists():
            try:
                pid = int(args.pid_file.read_text().strip())
            except Exception:
                pass

        if pid is None or not is_pid_alive(pid):
            print(f"[monitor-{args.task_id}] PID={pid} 不存在, 退出 monitor", flush=True)
            return

        cur_epoch = parse_current_epoch(args.log_file)
        for ck_epoch, util_thr, coll_thr in zip(args.check_epochs, args.util_kill_below, args.collision_kill_above):
            if ck_epoch in fired:
                continue
            if cur_epoch < ck_epoch:
                continue
            print(f"[monitor-{args.task_id}] 触发 epoch {ck_epoch} 检查 (cur_epoch={cur_epoch})", flush=True)
            fired.add(ck_epoch)

            # 等 ckpt 落盘
            time.sleep(10)
            ckpt = find_latest_ckpt(args.ckpt_dir)
            if ckpt is None:
                print(f"[monitor-{args.task_id}] WARN: 没找到 ckpt, 跳过 epoch {ck_epoch} 检查", flush=True)
                continue
            try:
                metrics = compute_util_and_collision(ckpt, args.assignment_mode, args.c_k_min, args.c_k_max)
            except Exception as e:
                print(f"[monitor-{args.task_id}] ERROR compute_util: {e}", flush=True)
                continue

            print(f"[monitor-{args.task_id}] epoch {ck_epoch} metrics: {metrics}", flush=True)

            should_kill = False
            reason_parts = []
            if metrics["L0_utilization"] < util_thr:
                should_kill = True
                reason_parts.append(f"L0_utilization={metrics['L0_utilization']:.4f} < {util_thr}")
            if metrics["collision_rate"] > coll_thr:
                should_kill = True
                reason_parts.append(f"collision_rate={metrics['collision_rate']:.4f} > {coll_thr}")

            if should_kill:
                reason = " | ".join(reason_parts)
                print(f"[monitor-{args.task_id}] 🔴 KILL: {reason}", flush=True)
                verdict_path = REPO / "verdicts" / f"task{args.task_id}_early_stop_result.md"
                write_verdict_stub(args.task_id, verdict_path, reason, metrics, ck_epoch)
                try:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(3)
                    if is_pid_alive(pid):
                        os.kill(pid, signal.SIGKILL)
                except Exception as e:
                    print(f"[monitor-{args.task_id}] kill error: {e}", flush=True)
                return

            print(f"[monitor-{args.task_id}] ✅ epoch {ck_epoch} 通过, 继续", flush=True)

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()