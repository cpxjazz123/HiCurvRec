#!/usr/bin/env python3
"""Task 304 真版: ABCD via Hydra pipeline_launcher"""
import sys, os, json, time
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task46_prob_vs_rank'
os.makedirs(OUT_DIR, exist_ok=True)

CKPT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task19_aq_s3/checkpoints/checkpoint_epoch=000_step=003900.ckpt'
SID_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'
DATA_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys'

K = 10
N_BATCH = 200
N_LAYERS = 4


def main():
    print('=' * 70)
    print('Task 304 真版: ABCD via Hydra')
    print('=' * 70)

    # Use pipeline_launcher like src.inference does
    from omegaconf import OmegaConf
    from src.utils.launcher_utils import pipeline_launcher

    cfg = OmegaConf.create({
        'ckpt_path': CKPT,
        'semantic_id_path': SID_PATH,
        'data_dir': DATA_DIR,
        'num_hierarchies': 4,
        'batch_size_per_device': 32,
        'sequence_length': 20,
        'seed': 42,
        'trainer': {
            'accelerator': 'gpu',
            'devices': 1,
            'strategy': 'auto',
        },
        'task_name': 'tiger_inference_flat',
        'paths': {
            'root': '/home/wlia0047/ar57/wenyu/GeneRec/GRID',
            'output_dir': '/tmp/task46_real',
        },
    })
    # Merge with base config (mirror what src.train does)
    base_cfg = OmegaConf.load('/home/wlia0047/ar57/wenyu/GeneRec/GRID/configs/inference.yaml')
    base_cfg.task_name = 'tiger_inference_flat'
    # Resolve experiment config
    exp_cfg = OmegaConf.load('/home/wlia0047/ar57/wenyu/GeneRec/GRID/configs/experiment/tiger_inference_flat.yaml')
    cfg = OmegaConf.merge(base_cfg, exp_cfg, cfg)

    print('\n[1] 启动 pipeline_launcher...')
    with pipeline_launcher(cfg) as pm:
        model = pm.model
        datamodule = pm.datamodule
        trainer = pm.trainer
        print(f'  model type: {type(model).__name__}')
        print(f'  num_hierarchies: {getattr(model, "num_hierarchies", "?")}')
        print(f'  top_k_for_generation: {getattr(model, "top_k_for_generation", "?")}')

        # Manually run predict using the test dataloader
        print(f'\n[2] 跑真实推断 (N_BATCH={N_BATCH})...')
        model.eval()
        model.freeze()
        device = next(model.parameters()).device
        print(f'  device: {device}')

        datamodule.setup('test')
        test_loader = datamodule.test_dataloader()
        print(f'  test loader: {len(test_loader)} batches')

        abcd_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        rank_shift_records = []
        n_users = 0
        n_full_hit = 0
        n_ablate_hit = 0
        t0 = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= N_BATCH:
                    break
                try:
                    # Move batch to device
                    if hasattr(batch, 'to'):
                        batch_dev = batch.to(device)
                    else:
                        batch_dev = batch

                    # predict_step uses model.generate()
                    out = model.predict_step(batch_dev, batch_idx)
                    preds = out.predictions if hasattr(out, 'predictions') else out
                    if isinstance(preds, torch.Tensor):
                        preds_cpu = preds.cpu()
                    else:
                        preds_cpu = torch.tensor(preds)

                    top_k = getattr(model, 'top_k_for_generation', K)
                    if preds_cpu.dim() == 2 and preds_cpu.shape[0] % top_k == 0:
                        preds_cpu = preds_cpu.reshape(-1, top_k, N_LAYERS).long()

                    # Get target
                    target_sids = None
                    for key in ('target_sids', 'fut_ids', 'labels', 'future_ids', 'label_sids'):
                        if hasattr(batch, key):
                            target_sids = getattr(batch, key)
                            break
                        if isinstance(batch, dict) and key in batch:
                            target_sids = batch[key]
                            break

                    if target_sids is None:
                        # Try nested access on SequentialModuleLabelData
                        if hasattr(batch, 'label_data') and batch.label_data is not None:
                            label_data = batch.label_data
                            if hasattr(label_data, 'labels'):
                                # labels is dict
                                for lk, lv in label_data.labels.items():
                                    target_sids = lv
                                    break
                        elif isinstance(batch, dict) and 'label_data' in batch and batch['label_data'] is not None:
                            label_data = batch['label_data']
                            if hasattr(label_data, 'labels'):
                                for lk, lv in label_data.labels.items():
                                    target_sids = lv
                                    break

                    if target_sids is None:
                        if batch_idx == 0:
                            keys = list(batch.keys()) if hasattr(batch, 'keys') else dir(batch)
                            print(f'  batch 0 keys: {keys[:20]}')
                        continue

                    if isinstance(target_sids, torch.Tensor):
                        target_sids = target_sids.cpu()
                    else:
                        target_sids = torch.tensor(target_sids)

                    if target_sids.dim() == 3:
                        target_sids = target_sids.squeeze(0)
                    if target_sids.dim() == 1:
                        target_sids = target_sids.reshape(-1, N_LAYERS)
                    target_sids = target_sids.long()

                    B = min(preds_cpu.shape[0], target_sids.shape[0])
                    if B == 0:
                        continue

                    for b in range(B):
                        tgt = target_sids[b]
                        pred = preds_cpu[b]
                        tgt_tup = tuple(int(x) for x in tgt.tolist())
                        pred_tuples = [tuple(int(x) for x in p.tolist()) for p in pred]

                        full_hit = tgt_tup in pred_tuples

                        tgt_abl = (tgt_tup[0], tgt_tup[1], tgt_tup[3])
                        ablate_hit = any(p[0] == tgt_abl[0] and p[1] == tgt_abl[1] and p[3] == tgt_abl[3]
                                         for p in pred_tuples)

                        if full_hit and ablate_hit:
                            abcd_counts['A'] += 1
                        elif full_hit and not ablate_hit:
                            abcd_counts['B'] += 1
                        elif not full_hit and ablate_hit:
                            abcd_counts['C'] += 1
                        else:
                            abcd_counts['D'] += 1

                        rank_full = next((i for i, p in enumerate(pred_tuples) if p == tgt_tup), None)
                        rank_abl = next((i for i, p in enumerate(pred_tuples)
                                         if p[0] == tgt_abl[0] and p[1] == tgt_abl[1] and p[3] == tgt_abl[3]), None)
                        if rank_full is not None and rank_abl is not None:
                            rank_shift_records.append(rank_abl - rank_full)

                        if full_hit:
                            n_full_hit += 1
                        if ablate_hit:
                            n_ablate_hit += 1

                    n_users += B
                    if batch_idx % 10 == 0:
                        elapsed = time.time() - t0
                        print(f'  [{batch_idx:3d}] users={n_users}, R@10_full={n_full_hit/max(1,n_users):.3f}, '
                              f'R@10_abl={n_ablate_hit/max(1,n_users):.3f}, ABCD={abcd_counts}')

                except Exception as e:
                    if batch_idx < 3:
                        import traceback
                        print(f'  batch {batch_idx} error: {type(e).__name__}: {e}')
                        traceback.print_exc()
                    continue

        elapsed = time.time() - t0
        print(f'\n[3] 完成: n_users={n_users}, elapsed={elapsed:.1f}s')
        print(f'  R@10 (full SID 4-digit): {n_full_hit/max(1,n_users):.4f}')
        print(f'  R@10 (ablate L3): {n_ablate_hit/max(1,n_users):.4f}')

        total = sum(abcd_counts.values())
        pct = {k: v/max(1,total) for k, v in abcd_counts.items()}
        print(f'  ABCD %: A={pct["A"]:.3f}, B={pct["B"]:.3f}, C={pct["C"]:.3f}, D={pct["D"]:.3f}')

        rank_shift = np.array(rank_shift_records) if rank_shift_records else np.array([])
        rs_stats = {
            'mean': float(rank_shift.mean()) if len(rank_shift) else None,
            'std': float(rank_shift.std()) if len(rank_shift) else None,
            'n': len(rank_shift),
            'positive': int((rank_shift > 0).sum()) if len(rank_shift) else 0,
            'negative': int((rank_shift < 0).sum()) if len(rank_shift) else 0,
        }

        out = {
            'ckpt': CKPT,
            'sid_path': SID_PATH,
            'n_users': n_users,
            'K': K,
            'elapsed_sec': elapsed,
            'R_at_10_full': n_full_hit / max(1, n_users),
            'R_at_10_ablate_L3': n_ablate_hit / max(1, n_users),
            'ABCD_counts': abcd_counts,
            'ABCD_pct': pct,
            'rank_shift_stats': rs_stats,
            'note': 'real TIGER inference on task19_aq_s3 ckpt via Hydra pipeline_launcher',
        }
        with open(os.path.join(OUT_DIR, 'abcd_real_tiger.json'), 'w') as f:
            json.dump(out, f, indent=2)

        with open(os.path.join(OUT_DIR, 'verdict_real_tiger.md'), 'w') as f:
            f.write('# Task 304 Verdict (真 TIGER 版)\n\n')
            f.write(f'数据集: Toys, K={K}, n_users={n_users}, 模型: task19_aq_s3/best_tiger_step3900.ckpt\n\n')
            f.write('## ABCD buckets (L3 digit 真实贡献)\n\n')
            f.write('| Bucket | Count | % | 含义 |\n')
            f.write('|--------|-------|---|------|\n')
            f.write(f'| A | {abcd_counts["A"]} | {pct["A"]:.3f} | full ∧ ablate 命中 → L3 不影响 |\n')
            f.write(f'| **B** | **{abcd_counts["B"]}** | **{pct["B"]:.3f}** | **full ∧ ¬ablate → L3 必要** |\n')
            f.write(f'| **C** | **{abcd_counts["C"]}** | **{pct["C"]:.3f}** | **¬full ∧ ablate → L3 噪声** |\n')
            f.write(f'| D | {abcd_counts["D"]} | {pct["D"]:.3f} | 都未命中 |\n\n')
            f.write(f'## 命中率\n\n')
            f.write(f'- R@10 (full SID): **{n_full_hit/max(1,n_users):.4f}**\n')
            f.write(f'- R@10 (ablate L3): {n_ablate_hit/max(1,n_users):.4f}\n')
            f.write(f'- 差异 (full - ablate): {n_full_hit/max(1,n_users) - n_ablate_hit/max(1,n_users):+.4f}\n\n')
            f.write(f'## Rank shift (rank_abl - rank_full, 双命中)\n\n')
            f.write(f'- n={rs_stats["n"]}, mean={rs_stats["mean"]}, std={rs_stats["std"]}\n')
            f.write(f'- L3 改善排名: {rs_stats["positive"]}, L3 损伤排名: {rs_stats["negative"]}\n\n')
            f.write('## 判读\n\n')
            b_pct, c_pct = pct['B'], pct['C']
            if b_pct > 0.05 and b_pct > c_pct:
                f.write(f'✅ L3 提供正向增量 (B={b_pct:.3f} > C={c_pct:.3f})\n')
            elif c_pct > b_pct:
                f.write(f'⚠️ L3 主要作为噪声 (C={c_pct:.3f} > B={b_pct:.3f})\n')
            else:
                f.write(f'➡️ L3 边际贡献接近零 (B={b_pct:.3f} ≈ C={c_pct:.3f})\n')

        print(f'\n[产物] {OUT_DIR}/abcd_real_tiger.json')
        print(f'[产物] {OUT_DIR}/verdict_real_tiger.md')


import numpy as np
import torch

if __name__ == '__main__':
    main()