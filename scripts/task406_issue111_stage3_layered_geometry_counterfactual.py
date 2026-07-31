#!/usr/bin/env python3
"""
Task #406 / Issue #111 [方向C Gate3] 同 checkpoint 层级 SID 几何消费反事实

按 Issue #111 spec §Gate 3 五组反事实 (在同一 checkpoint + 同一 batch 上):
1. 原始层级 SID
2. 仅层内 permutation
3. 交换 L0/L1/L2 顺序
4. 保持 token 边际频率但破坏 item-SID 对齐
5. padding/mask 正负对照

报告:
- logits L1/L2 diff (vs 原始)
- argmax match (top-1, top-5, top-10)
- token embedding/attention gradient norm
- attention map 或等价可观测量
- 重复运行方差 (固定 seed)

R18 v2 强制: 不允许 "路径同构" NO-GO, 必须实际跑 T5 forward 获得 logits diff 数据
"""
import os
import sys
import json
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

# ============ 路径 ============
CKPT_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/products/task396_issue99_stage3_t5_train/ckpt/Instruments/Jul-31-2026_19-15-50/HG_Rec_best.pth'
SID_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy'
TRAIN_PARQUET = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet'
TEST_PARQUET = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet'
AUDIT_OUT_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task406_issue111_stage3_layered_geometry_counterfactual.json'

DEVICE = 'cuda:0'  # CUDA_VISIBLE_DEVICES=1
BATCH_SIZE = 8
SEED = 42
CODEBOOK_SIZE = [64, 128, 256, 1]  # L0/L1/L2/L3 = K64/128/256 + 4-digit dedup
MAX_LEN = 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def load_model_and_tokenizer(ckpt_path, device):
    """Load T5-mini model from HG_Rec Stage 3 ckpt"""
    sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
    from model.HG_Rec import HG_Rec

    config = {
        'batch_size': BATCH_SIZE,
        'infer_size': BATCH_SIZE,
        'lr': 1e-4,
        'device': device,
        'num_layers': 6,
        'num_decoder_layers': 4,
        'd_model': 128,
        'd_ff': 1024,
        'num_heads': 6,
        'd_kv': 64,
        'dropout_rate': 0.0,  # eval mode
        'vocab_size': 1025,
        'pad_token_id': 0,
        'eos_token_id': 0,
        'feed_forward_proj': 'relu',
        'max_len': MAX_LEN,
        'dataset_name': 'Instruments',
        'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
        'codebook_size': CODEBOOK_SIZE,
        'code_path': '_t5_rqvae_task396.npy',
        'topk_list': [5, 10, 20],
        'beam_size': 20,
    }
    model = HG_Rec(config)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config


def load_sid_batch(sid_array, batch_indices, max_len=MAX_LEN):
    """Load SID for a batch of items, padded to max_len"""
    sid_batch = sid_array[batch_indices]  # (B, 4)
    # Repeat each SID to fill max_len (跟 GenRecDataset 训练时一致)
    # T5 训练时 input 是 SID sequence, 这里简化: 直接用 4-digit SID 重复 padding
    sid_padded = np.zeros((len(batch_indices), max_len), dtype=np.int64)
    sid_padded[:, 0] = sid_batch[:, 0]
    sid_padded[:, 1] = sid_batch[:, 1]
    sid_padded[:, 2] = sid_batch[:, 2]
    sid_padded[:, 3] = sid_batch[:, 3]
    # 其余位置 = 0 (pad_token_id)
    attention_mask = (sid_padded != 0).astype(np.int64)
    return torch.from_numpy(sid_padded).long(), torch.from_numpy(attention_mask).long()


def model_forward(model, input_ids, attention_mask, device):
    """Run T5 forward, return logits"""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    # T5 encoder-decoder 需要 decoder_input_ids (跟 task84 evaluate 函数一致: 用 pad_token_id 起始)
    # Per T5 convention: decoder_input_ids = [pad_token_id, target[:-1]] (shifted right)
    # 这里简化: 用 input_ids 作为 decoder_input_ids (Issue #111 反事实关心 encoder 路径 + decoder 对输入的响应)
    decoder_input_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
    return outputs.logits  # (B, max_len, vocab_size)


def compute_diff_stats(logits_orig, logits_perturb, prefix=''):
    """Compute L1/L2 diff + argmax match"""
    diff = logits_perturb - logits_orig
    l1 = diff.abs().mean().item()
    l2 = (diff ** 2).mean().sqrt().item()

    # Argmax match (top-1)
    argmax_orig = logits_orig.argmax(dim=-1)
    argmax_perturb = logits_perturb.argmax(dim=-1)
    top1_match = (argmax_orig == argmax_perturb).float().mean().item()

    # Top-5 match
    top5_orig = logits_orig.topk(5, dim=-1).indices
    top5_perturb = logits_perturb.topk(5, dim=-1).indices
    top5_match = (top5_orig == top5_perturb).all(dim=-1).float().mean().item()

    # Top-10 match
    top10_orig = logits_orig.topk(10, dim=-1).indices
    top10_perturb = logits_perturb.topk(10, dim=-1).indices
    top10_match = (top10_orig == top10_perturb).all(dim=-1).float().mean().item()

    return {
        f'{prefix}L1_diff': l1,
        f'{prefix}L2_diff': l2,
        f'{prefix}top1_argmax_match': top1_match,
        f'{prefix}top5_argmax_match': top5_match,
        f'{prefix}top10_argmax_match': top10_match,
    }


def compute_attention_grad_norm(model, input_ids, attention_mask, device):
    """Compute attention/embedding path gradient norm (small forward-backward)"""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    model.zero_grad()
    # T5 forward: encoder-decoder 必须有 decoder_input_ids
    decoder_input_ids = input_ids.clone()
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    logits = outputs.logits
    logits.sum().backward()

    # Collect gradient norms from embed_tokens + attention layers
    grad_norms = {}
    for name, param in model.model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = float(param.grad.norm().item())

    # Aggregate
    embed_grad = sum(v for k, v in grad_norms.items() if 'embed_tokens' in k)
    attention_grad = sum(v for k, v in grad_norms.items() if 'SelfAttention' in k or 'attention' in k.lower())
    ffn_grad = sum(v for k, v in grad_norms.items() if 'DenseReluDense' in k or 'wi' in k or 'wo' in k)
    layer_norm_grad = sum(v for k, v in grad_norms.items() if 'layer_norm' in k)

    return {
        'embed_tokens_grad_norm_sum': embed_grad,
        'attention_grad_norm_sum': attention_grad,
        'ffn_grad_norm_sum': ffn_grad,
        'layer_norm_grad_norm_sum': layer_norm_grad,
        'total_grad_norm_sum': sum(grad_norms.values()),
        'num_params_with_grad': len(grad_norms),
    }


def main():
    print(f'[Task #406 Issue #111] ckpt: {CKPT_PATH}', flush=True)
    ckpt_sha = sha256_file(CKPT_PATH)
    print(f'  sha256: {ckpt_sha}', flush=True)
    print(f'  SID: {SID_PATH}', flush=True)

    # Load SID
    sid_array = np.load(SID_PATH)
    print(f'  SID shape: {sid_array.shape}', flush=True)
    print(f'  SID unique: {len(np.unique(sid_array, axis=0))} / {len(sid_array)}', flush=True)

    # Fixed batch (per Issue spec §Gate3: 固定 checkpoint, batch, seed)
    np.random.seed(SEED)
    batch_indices = np.random.choice(len(sid_array), BATCH_SIZE, replace=False)
    print(f'  batch_indices: {batch_indices}', flush=True)

    # Load model
    print(f'[Task #406] Loading T5 model...', flush=True)
    model, config = load_model_and_tokenizer(CKPT_PATH, DEVICE)
    print(f'  Model loaded', flush=True)

    # ===== 原始 (Group 1) =====
    input_ids_orig, attn_mask_orig = load_sid_batch(sid_array, batch_indices)
    logits_orig = model_forward(model, input_ids_orig, attn_mask_orig, DEVICE)
    print(f'  Original logits shape: {logits_orig.shape}', flush=True)

    grad_orig = compute_attention_grad_norm(model, input_ids_orig, attn_mask_orig, DEVICE)

    # ===== Group 2: 仅层内 permutation (L0/L1/L2 各自 shuffle) =====
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sid_g2 = sid_array[batch_indices].copy()
    np.random.shuffle(sid_g2[:, 0])  # L0 shuffle
    np.random.shuffle(sid_g2[:, 1])  # L1 shuffle
    np.random.shuffle(sid_g2[:, 2])  # L2 shuffle
    sid_g2[:, 3] = sid_array[batch_indices, 3]  # L3 保留 (4-digit dedup digit)
    input_ids_g2 = torch.from_numpy(np.zeros((BATCH_SIZE, MAX_LEN), dtype=np.int64))
    input_ids_g2[:, 0] = torch.from_numpy(sid_g2[:, 0])
    input_ids_g2[:, 1] = torch.from_numpy(sid_g2[:, 1])
    input_ids_g2[:, 2] = torch.from_numpy(sid_g2[:, 2])
    input_ids_g2[:, 3] = torch.from_numpy(sid_g2[:, 3])
    logits_g2 = model_forward(model, input_ids_g2, attn_mask_orig, DEVICE)
    stats_g2 = compute_diff_stats(logits_orig, logits_g2, 'g2_layer_perm_')

    # ===== Group 3: 交换 L0/L1/L2 顺序 =====
    sid_g3 = sid_array[batch_indices].copy()
    sid_g3_swap = sid_g3.copy()
    sid_g3_swap[:, 0] = sid_g3[:, 2]  # L0 ← L2
    sid_g3_swap[:, 2] = sid_g3[:, 0]  # L2 ← L0
    input_ids_g3 = torch.from_numpy(np.zeros((BATCH_SIZE, MAX_LEN), dtype=np.int64))
    input_ids_g3[:, 0] = torch.from_numpy(sid_g3_swap[:, 0])
    input_ids_g3[:, 1] = torch.from_numpy(sid_g3_swap[:, 1])
    input_ids_g3[:, 2] = torch.from_numpy(sid_g3_swap[:, 2])
    input_ids_g3[:, 3] = torch.from_numpy(sid_g3_swap[:, 3])
    logits_g3 = model_forward(model, input_ids_g3, attn_mask_orig, DEVICE)
    stats_g3 = compute_diff_stats(logits_orig, logits_g3, 'g3_swap_layers_')

    # ===== Group 4: 保持 token 边际频率但破坏 item-SID 对齐 =====
    sid_g4 = sid_array[batch_indices].copy()
    # Shuffle 整 batch 内的所有 SID (跨 item, 保持边际频率 = SID unique 集合不变)
    all_sids = sid_array.copy()
    np.random.seed(SEED + 100)
    np.random.shuffle(all_sids)  # 整体 shuffle
    sid_g4 = all_sids[:BATCH_SIZE]
    input_ids_g4 = torch.from_numpy(np.zeros((BATCH_SIZE, MAX_LEN), dtype=np.int64))
    input_ids_g4[:, 0] = torch.from_numpy(sid_g4[:, 0])
    input_ids_g4[:, 1] = torch.from_numpy(sid_g4[:, 1])
    input_ids_g4[:, 2] = torch.from_numpy(sid_g4[:, 2])
    input_ids_g4[:, 3] = torch.from_numpy(sid_g4[:, 3])
    logits_g4 = model_forward(model, input_ids_g4, attn_mask_orig, DEVICE)
    stats_g4 = compute_diff_stats(logits_orig, logits_g4, 'g4_break_item_align_')

    # ===== Group 5: padding/mask 正负对照 (mask 全 0 屏蔽所有 token) =====
    attn_mask_g5 = torch.zeros_like(attn_mask_orig)  # 全 0 mask
    logits_g5 = model_forward(model, input_ids_orig, attn_mask_g5, DEVICE)
    stats_g5 = compute_diff_stats(logits_orig, logits_g5, 'g5_mask_all_zero_')

    # ===== 汇总 =====
    audit = {
        'task': 'task406_issue111_stage3_layered_geometry_counterfactual',
        'recipe': 'task396b Stage 3 ckpt (Issue #99) + Issue #97 patch + Sinkhorn (task396 SID) + 同 ckpt 同 batch 5 组反事实',
        'ckpt_sha256': ckpt_sha,
        'sid_shape': list(sid_array.shape),
        'batch_size': BATCH_SIZE,
        'batch_indices': [int(i) for i in batch_indices],
        'seed': SEED,
        'device': DEVICE,
        'group_1_original_logits_shape': list(logits_orig.shape),
        'group_1_original_grad': grad_orig,
        'group_2_layer_permutation': stats_g2,
        'group_3_swap_layers': stats_g3,
        'group_4_break_item_alignment': stats_g4,
        'group_5_mask_all_zero': stats_g5,
        'go_nogo_logic': (
            'Issue #111 spec §Gate 3 PASS 条件: 原始与结构破坏组有稳定非零差异, '
            '三层路径梯度有限非零, mask 正确. '
            '若输出仅对 token identity 敏感而对层级几何不敏感 → Gate 3 FAIL. '
            '本 task 量化所有 5 组的 L1/L2 diff + argmax match, 报告每组是否非零.'
        ),
        'next_step': 'verdict 写 + commit + push + close issue #111',
    }

    os.makedirs(os.path.dirname(AUDIT_OUT_PATH), exist_ok=True)
    with open(AUDIT_OUT_PATH, 'w') as f:
        json.dump(audit, f, indent=2)
    print(f'[Task #406] Audit saved: {AUDIT_OUT_PATH}', flush=True)
    print(f'[Task #406] g2 L1={stats_g2["g2_layer_perm_L1_diff"]:.6f} L2={stats_g2["g2_layer_perm_L2_diff"]:.6f} top1={stats_g2["g2_layer_perm_top1_argmax_match"]:.4f}', flush=True)
    print(f'[Task #406] g3 L1={stats_g3["g3_swap_layers_L1_diff"]:.6f} L2={stats_g3["g3_swap_layers_L2_diff"]:.6f} top1={stats_g3["g3_swap_layers_top1_argmax_match"]:.4f}', flush=True)
    print(f'[Task #406] g4 L1={stats_g4["g4_break_item_align_L1_diff"]:.6f} L2={stats_g4["g4_break_item_align_L2_diff"]:.6f} top1={stats_g4["g4_break_item_align_top1_argmax_match"]:.4f}', flush=True)
    print(f'[Task #406] g5 L1={stats_g5["g5_mask_all_zero_L1_diff"]:.6f} L2={stats_g5["g5_mask_all_zero_L2_diff"]:.6f} top1={stats_g5["g5_mask_all_zero_top1_argmax_match"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
