"""DDP RQ-VAE training on pre-computed Amazon-2023 Instruments embeddings.

Loads item_emb.npy (24587, 768) and trains RqVae on 4 GPUs via torchrun.

执行流程按 Step1-Step12 拆分（详见 modules/step_checks.py）:
  Step1   配置硬编码不变量检查
  Step2   数据集加载与有限性检查
  Step3   DDP / distributed 初始化
  Step4   模型结构与参数初始化检查
  Step5   optimizer 覆盖全部可训练参数
  Step6   单个 batch 取数与设备一致性检查
  Step7   RQ-VAE 前向 (encoder + n 层量化 + 残差 + 曲率传输)
  Step8   reconstruction + rqvae loss 汇总
  Step9   反向传播后所有可训练参数梯度非零
  Step10  optimizer.step() 至少更新一项参数
  Step11  curriculum step 同步曲率 c(t)
  Step12  checkpoint 落盘且非空

=== 2026-09-17 MIGRATED FROM curvature_base ===
源: /home/wlia0047/ar57/wenyu/GeneRec/curvature_base/train_rqvae_instruments.py (镜像于 curvature_RQ-VAE_iter5/)
迁移内容: 启用 USE_GEODESIC_MIDPOINT_COMMIT=True, USE_M2_INTRINSIC=True, USE_M3_TRANSPORT=True,
         COMMITMENT_WEIGHT=1.0 (HG-Rec default). 保留 cyclic c(t) curriculum 与 4 卡 DDP.
迁移目的: 修复 v400e 版 layer0 collapse (67/256 → 250+/256 codes).
源文件备份: curvature_RQ-VAE.py.v400e_final.bak (md5=4c4c8ad6...).
源 ckpt 备份: out/rqvae/instruments_v400e_pre_migration/rqvae_final.pt.

启动:
  cd /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE
  python3 curvature_RQ-VAE.py  (内部 spawn torchrun --nproc_per_node=4)
  或 torchrun --standalone --nproc_per_node=4 --master_port=50200 curvature_RQ-VAE.py

R36/R47: items are precomputed by stage1_GeneEmbedding using sentence-t5-xxl.
R42: 必须 torchrun --nproc_per_node=4 (DDP 4 卡) 或兼容单卡 fallback.
R30/R43: 超参硬编码, 无 CLI 数值超参.
R53 v3.8: 路径和训练超参全部从 curvature_config.py 硬编码导入, 不依赖 cwd 或环境变量.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# R47 imports — RQ-VAE-Recommender modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.rqvae import RqVae
from data.schemas import SeqBatch
from modules.step_checks import (
    check_step1_config,
    check_step2_dataset,
    check_step3_distributed,
    check_step5_optimizer,
    check_step6_batch,
    check_step9_backward,
    check_step10_update,
    check_step11_curvature,
    check_step12_checkpoint,
)
from modules.sid_quality import (
    build_sids_for_corpus,
    evaluate_sid_quality_full,
    should_early_stop,
    format_metrics as format_sid_metrics,
    write_quality_report,
)
from modules.step_checks import check_step0_clean_output

# R52/R53: 路径相对 cwd, 从 curvature_config.py 硬编码导入
from curvature_config import ITEM_EMB_NPY as EMB_NPY, RQVAE_OUT_DIR as OUT_DIR


# === 超参 (硬编码 R30/R43) ===
SEED = 42

INPUT_DIM = 768
HIDDEN_DIMS = [512, 256, 128]
EMBED_DIM = 32
CODEBOOK_SIZE = 256
N_LAYERS = 3
COMMITMENT_WEIGHT = 1.0  # HG-Rec default beta=1.0 — commitment/codebook loss 等权

# === 训练硬约束 (R30/R43): 100k global steps ===
MAX_GLOBAL_STEPS = 100_000       # 100k steps ≈ 7 分钟@4卡
CKPT_EVERY = 10_000              # 每 10k 全球 step 保存一份 ckpt

# codebook 健康检查: 层 unique code 数 < CODEBOOK_SIZE*该阈值 → 打印 [CODEBOOK WARNING]
CODEBOOK_COLLAPSE_THRESHOLD = 0.10

# 当前启用：cyclic curvature、M2/M3；所有层使用 Möbius residual subtraction。
C_CYCLIC_MIN = 0.3
C_CYCLIC_MAX = 1.0
C_CYCLIC_PERIOD = 50_000
GRAD_CLIP_NORM = 1.0
MIDPOINT_LAYER_MASK = [False, False, False]

# === 加速调参 (硬编码, R36 加速 OK) ===
BATCH_SIZE = 640                 # per-GPU batch (4 卡 DDP, 总 batch = 2560)
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
COMPILE = False


class EmbeddingDataset(Dataset):
    def __init__(self, emb_path: str):
        arr = np.load(emb_path).astype(np.float32)
        self.embeddings = torch.from_numpy(arr)
        check_step2_dataset(self, INPUT_DIM)
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            print(
                f"[Step2] loaded {emb_path}: shape={tuple(self.embeddings.shape)}, "
                f"dtype={self.embeddings.dtype}",
                flush=True,
            )

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


def collate_items(batch):
    return torch.stack(batch, dim=0)


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_ckpt(model, optimizer, global_step, out_dir, tag):
    """rank 0 only: 保存 model.module.state_dict() + optimizer + step."""
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    path = os.path.join(out_dir, f"rqvae_{tag}.pt")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }
    torch.save(state, path)
    print(f"[Step12] saved {path} (step={global_step})", flush=True)
    check_step12_checkpoint(path)


def _delete_previous_artifact(out_dir: str, prev_tag: str) -> None:
    """rank 0 only: 删除上一个训练节点的 3 类产物 (ckpt + sids + quality).

    prev_tag 是上一个节点的 tag (例如 'step10000' / 'final'), 必须存在; 若不存在
    则直接 raise (不允许 fallback). 任何删除失败同样 raise.
    """
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    if not prev_tag:
        raise ValueError("_delete_previous_artifact: prev_tag 不能为空")
    targets = (
        os.path.join(out_dir, f"rqvae_{prev_tag}.pt"),
        os.path.join(out_dir, f"sids_{prev_tag}.npy"),
        os.path.join(out_dir, f"quality_{prev_tag}.json"),
    )
    for path in targets:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"_delete_previous_artifact: 期望的产物不存在, 无法清理: {path}"
            )
        try:
            os.unlink(path)
        except OSError as exc:
            raise RuntimeError(
                f"_delete_previous_artifact: 删除失败 {path}: {exc}"
            ) from exc
    print(
        f"[Step12] deleted previous artifacts with tag={prev_tag} "
        f"(rqvae_{prev_tag}.pt + sids_{prev_tag}.npy + quality_{prev_tag}.json)",
        flush=True,
    )


def _build_seq_batch(x: torch.Tensor, device: torch.device) -> SeqBatch:
    bsz = x.shape[0]
    return SeqBatch(
        user_ids=torch.zeros(bsz, dtype=torch.long, device=device),
        ids=torch.arange(bsz, dtype=torch.long, device=device),
        ids_fut=torch.zeros(bsz, dtype=torch.long, device=device),
        x=x,
        x_fut=x,
        seq_mask=torch.ones(bsz, dtype=torch.long, device=device),
    )


def main():
    # ---------- Step1 配置不变量 ----------
    check_step1_config(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        codebook_size=CODEBOOK_SIZE,
        n_layers=N_LAYERS,
        c_min=C_CYCLIC_MIN,
        c_max=C_CYCLIC_MAX,
        c_period=C_CYCLIC_PERIOD,
        sk_eps=0.05,
        sk_iters=3,
    )

    # ---------- Step3 DDP 初始化 ----------
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    check_step3_distributed(rank, world_size, local_rank)
    if rank == 0:
        print(f"[Step3] DDP rank={rank}/{world_size} local_rank={local_rank}", flush=True)

    # ---------- Step0 每次训练前清空旧产物 (rank 0 删除 + 其他 rank barrier) ----------
    deleted = check_step0_clean_output(OUT_DIR, rank=rank, barrier=True)
    if rank == 0:
        print(f"[Step0] cleaned {deleted} stale artifact(s) from {OUT_DIR}", flush=True)

    torch.manual_seed(SEED + rank)
    np.random.seed(SEED + rank)

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        print(
            f"=== RQ-VAE Step1-Step12 训练循环 (world_size={world_size}) ===",
            flush=True,
        )
        print(f"[Step1] emb: {EMB_NPY}", flush=True)
        print(f"[Step1] out: {OUT_DIR}", flush=True)
        print(
            f"[Step1] per-GPU batch={BATCH_SIZE} | total={BATCH_SIZE*world_size} | "
            f"workers={NUM_WORKERS} | compile={COMPILE}",
            flush=True,
        )
        print(
            f"[Step1] hidden={HIDDEN_DIMS} embed={EMBED_DIM} codebook={CODEBOOK_SIZE} "
            f"layers={N_LAYERS}",
            flush=True,
        )
        print(
            f"[Step1] target: MAX_GLOBAL_STEPS={MAX_GLOBAL_STEPS} | ckpt every {CKPT_EVERY}",
            flush=True,
        )

    # ---------- Step2 数据集 ----------
    ds = EmbeddingDataset(EMB_NPY)
    n_items = len(ds)
    sampler = DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=True,
        seed=SEED, drop_last=True,
    )
    train_loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=True,
        collate_fn=collate_items,
    )

    if rank == 0:
        print(
            f"[Step2] {n_items} items | {len(train_loader)} batches/rank | "
            f"{len(train_loader)*world_size} batches/epoch",
            flush=True,
        )

    # ---------- Step4 模型 ----------
    # === iter5 v375: mixed-curvature product manifold H^{c1} x H^{c2} x S ===
    PER_LAYER_C_MIN = [0.3, 0.5, 0.7]
    PER_LAYER_C_MAX = [1.0, 1.2, 1.5]
    PER_LAYER_C_PERIOD = [50_000, 50_000, 50_000]
    # L0/L1 = poincare (hyp), L2 = sphere. 球面用于最后一层, 避免深层 commit 失真.
    PER_LAYER_MANIFOLD = ["poincare", "poincare", "sphere"]
    # === iter11 sweep: L0-only sk_eps Gini sweep knob ===
    # 默认 = [0.05, 0.05, 0.05] 等价旧行为. sweep 时只改 L0 (索引 0).
    # L0 sk_eps ↓ → Sinkhorn 越"硬" (近 argmax) → L0 Gini↑ (head/tail 形成)
    # L1/L2 锁死 0.05 (保留现有均匀化行为, 不让深层 collapse).
    PER_LAYER_SK_EPS = [0.001, 0.05, 0.05]
    model = RqVae(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        codebook_size=CODEBOOK_SIZE,
        codebook_kmeans_init=True,
        n_layers=N_LAYERS,
        commitment_weight=COMMITMENT_WEIGHT,
        sk_eps=0.05,
        sk_iters=3,
        c_cyclic_min=min(PER_LAYER_C_MIN),
        c_cyclic_max=max(PER_LAYER_C_MAX),
        c_cyclic_period=max(PER_LAYER_C_PERIOD),
        midpoint_layer_mask=MIDPOINT_LAYER_MASK,
        per_layer_c_min=PER_LAYER_C_MIN,
        per_layer_c_max=PER_LAYER_C_MAX,
        per_layer_c_period=PER_LAYER_C_PERIOD,
        per_layer_manifold=PER_LAYER_MANIFOLD,
        per_layer_sk_eps=PER_LAYER_SK_EPS,
    ).to(device)

    if COMPILE:
        model = torch.compile(model)

    if dist.is_available() and dist.is_initialized():
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )

    if rank == 0:
        n_params = sum(p.numel() for p in model.module.parameters())
        print(f"[Step4] params={n_params} | DDP wrapped", flush=True)

    # ---------- Step5 优化器 ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    check_step5_optimizer(optimizer, model.module)
    if rank == 0:
        print(f"[Step5] optimizer covers {len(optimizer.param_groups)} group(s)", flush=True)

    global_step = 0   # 全球 step (= 单卡 step × world_size, 用 all_reduce SUM 同步)
    t_start = time.time()
    last_log_t = t_start
    last_saved_tag: str | None = None  # 上一个 ckpt 的 tag, 用于新 ckpt 落盘后清理
    # 仅打印一次的首个 step 触发证据, 避免每 5s log 把 check marker 淹没
    step_first_pass_logged = {
        6: False, 7: False, 8: False, 9: False, 10: False, 11: False,
    }

    if rank == 0:
        print(
            f"[train] start, target={MAX_GLOBAL_STEPS} global steps (Step1-Step12 loop)",
            flush=True,
        )
    model.train()
    done = False
    while not done:
        sampler.set_epoch(global_step)
        for batch in train_loader:
            # ---------- Step6 单个 batch ----------
            x = batch.to(device, non_blocking=True)
            check_step6_batch(x, INPUT_DIM, device)
            if rank == 0 and not step_first_pass_logged[6]:
                print(f"[Step6] first batch ok shape={tuple(x.shape)} device={x.device}", flush=True)
                step_first_pass_logged[6] = True
            seq_batch = _build_seq_batch(x, device)

            # ---------- Step7 + Step8 前向与 loss ----------
            optimizer.zero_grad()
            t_step = time.time()
            out = model(seq_batch)
            loss = out.loss
            if rank == 0 and not step_first_pass_logged[7]:
                ids_shape = tuple(out.sem_ids_shape)
                print(f"[Step7] first forward ok sem_ids.shape={ids_shape}", flush=True)
                step_first_pass_logged[7] = True
            if rank == 0 and not step_first_pass_logged[8]:
                print(
                    f"[Step8] first loss ok loss={float(loss.item()):.4f} "
                    f"rl={float(out.reconstruction_loss.item()):.4f} "
                    f"vl={float(out.rqvae_loss.item()):.4f}",
                    flush=True,
                )
                step_first_pass_logged[8] = True

            # ---------- Step9 反向传播 ----------
            loss.backward()
            # HG-Rec fix: gradient clipping at norm 1.0 (实测稳定 +∞ norm 梯度, 防 NaN)
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            check_step9_backward(model.module)
            if rank == 0 and not step_first_pass_logged[9]:
                print(f"[Step9] first backward ok params_with_grad={sum(1 for p in model.module.parameters() if p.grad is not None)}", flush=True)
                step_first_pass_logged[9] = True

            # ---------- Step10 optimizer.step() ----------
            params_before = [
                parameter.detach().clone()
                for parameter in model.module.parameters()
                if parameter.requires_grad
            ]
            optimizer.step()
            check_step10_update(params_before, model.module)
            if rank == 0 and not step_first_pass_logged[10]:
                print(f"[Step10] first optimizer.step ok", flush=True)
                step_first_pass_logged[10] = True
            global_step += 1

            # 同步全球 step 计数 (rank 0 累加其他 rank 的 +1)
            step_tensor = torch.tensor([global_step], dtype=torch.long, device=device)
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            global_step_sync = int(step_tensor.item())

            # ---------- Step11 curriculum step 同步曲率 c(t) ----------
            _cur_model = model.module if hasattr(model, "module") else model
            _cur_model.set_curriculum_step(global_step_sync)
            check_step11_curvature(model.module, global_step_sync, min(PER_LAYER_C_MIN), max(PER_LAYER_C_MAX))
            if rank == 0 and not step_first_pass_logged[11]:
                curv_str = "/".join(f"{float(l.get_c().item()):.3f}" for l in model.module.layers)
                print(f"[Step11] first curvature sync step={global_step_sync} c=[{curv_str}]", flush=True)
                step_first_pass_logged[11] = True

            now = time.time()
            if rank == 0 and (now - last_log_t >= 5.0):
                elapsed = now - t_start
                ips_global = global_step_sync / elapsed
                ips_per_gpu = (global_step_sync / world_size) / elapsed
                eta_s = (
                    (MAX_GLOBAL_STEPS - global_step_sync) / ips_global
                    if ips_global > 0 else float("inf")
                )
                usage = [int(u) for u in out.per_layer_usage.tolist()]
                usage_str = "/".join(str(u) for u in usage)
                low_usage = [
                    li for li, u in enumerate(usage)
                    if u < CODEBOOK_SIZE * CODEBOOK_COLLAPSE_THRESHOLD
                ]
                curvs = [float(l.get_c().item()) for l in model.module.layers]
                curv_str = "/".join(f"{c:.3f}" for c in curvs)
                print(
                    f"  step {global_step_sync:6d}/{MAX_GLOBAL_STEPS} "
                    f"| loss={float(loss.item()):.4f} "
                    f"| rl={float(out.reconstruction_loss.item()):.4f} "
                    f"| vl={float(out.rqvae_loss.item()):.4f} "
                    f"| codes={usage_str}/{CODEBOOK_SIZE} "
                    f"| c={curv_str} "
                    f"| ips_g={ips_global:.1f} ips/rk={ips_per_gpu:.1f} "
                    f"| elapsed={elapsed:.1f}s | eta={eta_s:.1f}s",
                    flush=True,
                )
                if low_usage:
                    print(
                        f"  [CODEBOOK WARNING] layer {low_usage} usage={usage_str}/"
                        f"{CODEBOOK_SIZE} (< {CODEBOOK_COLLAPSE_THRESHOLD:.0%}), "
                        f"疑似 collapse, 建议终止检查",
                        flush=True,
                    )
                last_log_t = now

            # ---------- Step12 checkpoint 落盘 + SID 质量门控 + 上一个清理 ----------
            if global_step_sync >= MAX_GLOBAL_STEPS or (
                global_step_sync > 0 and global_step_sync % CKPT_EVERY == 0
            ):
                tag = f"step{global_step_sync}"
                if global_step_sync >= MAX_GLOBAL_STEPS:
                    tag = "final"
                save_ckpt(model, optimizer, global_step_sync, OUT_DIR, tag)
                dist.barrier()

                # ----- SID 质量门控 (R2 / CLAUDE.md): 每个 ckpt 落盘点评估 -----
                # 仅 rank 0 负责全量推理、写报告和生成停止状态；所有 rank 必须经过
                # 同一个 broadcast，避免 rank 0 提前 break、其他 rank 继续进入 collective。
                early_stop_flag = torch.zeros(1, dtype=torch.int64, device=device)
                if rank == 0:
                    try:
                        sids = build_sids_for_corpus(model, EMB_NPY, device)
                        sid_path = os.path.join(OUT_DIR, f"sids_{tag}.npy")
                        np.save(sid_path, sids)
                        metrics = evaluate_sid_quality_full(sids, EMB_NPY)
                        early_stopped, reason = should_early_stop(metrics)
                        print(
                            f"  [SID gate step={global_step_sync}] "
                            f"{format_sid_metrics(metrics)} "
                            f"| early_stop={early_stopped}"
                            f"{(' reason=' + reason) if reason else ''}",
                            flush=True,
                        )
                        write_quality_report(
                            os.path.join(OUT_DIR, f"quality_{tag}.json"),
                            step=global_step_sync,
                            metrics=metrics,
                            early_stopped=early_stopped,
                            reason=reason,
                        )
                        if early_stopped:
                            print(
                                f"[EARLY STOP step={global_step_sync}] "
                                f"{reason}; 停止训练, 已保存 {tag} ckpt + sids",
                                flush=True,
                            )
                            # 早停时也清理上一个节点的产物, 只保留当前 + quality
                            if last_saved_tag is not None:
                                _delete_previous_artifact(OUT_DIR, last_saved_tag)
                        early_stop_flag[0] = int(early_stopped)
                    except Exception as sid_exc:
                        # SID 评估失败不允许 fallback — 必须 raise 让用户看到
                        raise RuntimeError(
                            f"SID 质量评估在 step={global_step_sync} 失败"
                        ) from sid_exc

                dist.broadcast(early_stop_flag, src=0)
                if bool(early_stop_flag.item()):
                    done = True
                    break

                # 正常路径: 仅 rank 0 删上一个节点并更新本地 tag。
                if rank == 0:
                    if last_saved_tag is not None:
                        _delete_previous_artifact(OUT_DIR, last_saved_tag)
                    last_saved_tag = tag

            if global_step_sync >= MAX_GLOBAL_STEPS:
                done = True
                break

    if rank == 0:
        print(
            f"[train] done at global_step={global_step_sync}, "
            f"total time={(time.time()-t_start):.1f}s",
            flush=True,
        )


# === 单一 Python 入口: `python curvature_RQ-VAE.py` —— 0 CLI args, 全部 torchrun 旗标 hard-code.
import subprocess as _subprocess
import sys as _sys

_LAUNCHER = {
    "torchrun":   "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/torchrun",
    "script":     os.path.abspath(__file__),
    "nproc":      4,
    "master_port": 50200,
    "log":        "logs/train_migrated.log",
}


def _launch_via_torchrun():
    """Fork torchrun subprocess with hard-coded flags."""
    log_dir = os.path.dirname(_LAUNCHER["log"])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    cmd = [
        _LAUNCHER["torchrun"],
        "--standalone",
        f"--nproc_per_node={_LAUNCHER['nproc']}",
        f"--master_port={_LAUNCHER['master_port']}",
        _LAUNCHER["script"],
    ]
    with open(_LAUNCHER["log"], "wb") as fout:
        rc = _subprocess.call(cmd, stdout=fout, stderr=_subprocess.STDOUT)
    _sys.exit(rc)


if __name__ == "__main__":
    # Already a torchrun worker (RANK set) -> main directly.
    # Otherwise, fork torchrun.
    if "RANK" in os.environ:
        try:
            main()
        finally:
            cleanup_distributed()
    else:
        _launch_via_torchrun()
