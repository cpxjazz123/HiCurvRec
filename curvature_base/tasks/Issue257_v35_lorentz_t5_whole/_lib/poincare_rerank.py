"""Issue #238 (2026-08-10): Stage4 Hyperbolic Re-ranking — 在 beam search 后用 Poincaré 距离 rerank.

核心思路: 曲率不进 cross-entropy 路径(那是 #100 失败点),只进 post-generation rerank.
  - Stage3 保持 v18 base (matmul attention + CE generation)
  - Stage4 新增 curvature rerank: score_final = score_orig + alpha * R_geo
  - R_geo = -d_P(history_emb, candidate_emb), 鼓励 candidates 与 history 在 Poincaré 空间上接近

实现:
  - L0 codebook (64 entries × 32 dim 切空间) 提供 item embedding
  - exp_map_0(v) 把切空间映到 Poincaré 球 (Poincaré ball, ||·|| < 1)
  - d_P(u, v) = arcosh(1 + 2*||u-v||^2 / ((1-||u||^2)*(1-||v||^2))) / sqrt(c)
  - history_emb = mean_{l0 in history} exp_map_0(L0[l0])  (用户历史 centroid)
  - cand_emb = exp_map_0(L0[cand_l0])  (候选 item L0 位置)

复杂度: O(B * K * n_history) per user, K=beam_size=20, n_history ≈ 20, B=batch ≈ 32.
"""
import torch
import numpy as np


def exp_map_0(v, c):
    """把切空间向量 v 映到 Poincaré 球 (c > 0). exp_map_0^c(v) = tanh(sqrt(c)*||v||)*v / (sqrt(c)*||v||)
    v: (..., d). 返回 (..., d), ||result|| < 1.
    """
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sqrt_c = torch.sqrt(c).clamp_min(1e-3)
    factor = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)
    return factor * v


def poincare_distance_sq(u, v, c):
    """d_P(u, v) = arcosh(1 + 2*||u-v||^2 / ((1-||u||^2)*(1-||v||^2))) / sqrt(c).
    u: (..., L, d), v: (..., L_v, d). 返回 (..., L, L_v).
    """
    u_norm_sq = (u * u).sum(dim=-1, keepdim=True)  # (..., L, 1)
    v_norm_sq = (v * v).sum(dim=-1, keepdim=True).transpose(-1, -2)  # (..., 1, L_v)
    u_norm_sq = u_norm_sq.clamp(max=1.0 - 1e-5)
    v_norm_sq = v_norm_sq.clamp(max=1.0 - 1e-5)
    diff = u.unsqueeze(-2) - v.unsqueeze(-3)  # (..., L, L_v, d)
    diff_sq = (diff * diff).sum(dim=-1)  # (..., L, L_v)
    denom = (1.0 - u_norm_sq) * (1.0 - v_norm_sq)
    denom = denom.clamp_min(1e-10)
    x = 1.0 + 2.0 * diff_sq / denom
    x = x.clamp_min(1.0 + 1e-7)  # arcosh 定义域
    return torch.acosh(x) / torch.sqrt(c.clamp_min(1e-3))


def load_poincare_assets(stage2_ckpt_path, layer=0):
    """从 Stage2 ckpt 加载 L0 codebook (切空间) + final_cs[0] (c 值).

    Returns:
        codebook: (K_l, d_tangent) torch tensor, 切空间 L0
        c: scalar torch tensor, final_cs[layer] (Poincaré 球 c 值)
    """
    ckpt = torch.load(stage2_ckpt_path, map_location='cpu', weights_only=False)
    codebook = ckpt['model_state_dict'][f'vq_layers.{layer}.embeddings.weight']  # (K_l, d)
    final_cs = ckpt['final_cs'][layer]  # scalar
    c = torch.tensor(final_cs, dtype=codebook.dtype)
    return codebook, c


def extract_l0_token(token_id):
    """从 SID token id 提取 L0 (粗粒度) 层 token id.
    Token id 范围:
      1-64: L0 (layer 0)
      65-192: L1 (layer 1)
      193-448: L2 (layer 2)
      449: L3 (dedup)

    返回 L0 索引 0-63 (减 1). PAD (0) / L1 / L2 / L3 → 返回 -1 (mask, 不计入 history centroid).
    只用 L0 codebook (64 entries) 算 R_geo 时, 必须严格 mask 非 L0 token.
    """
    if 1 <= token_id <= 64:
        return token_id - 1  # 0-63
    return -1  # 其他全部 mask (PAD / L1 / L2 / L3)


def compute_geo_score_batch(history_input_ids, candidate_sid_seqs, codebook_t, c_t, device):
    """计算 R_geo per user-candidate pair.

    Args:
        history_input_ids: (B, L_hist) long, 用户历史 token ids (含 PAD/L0/L1/L2/L3)
        candidate_sid_seqs: (B, K, L_cand) long, 候选 SID 序列 (L0/L1/L2/L3 tokens)
        codebook_t: (K_l, d_tangent) torch, L0 codebook
        c_t: scalar torch, c 值

    Returns:
        R_geo: (B, K) float, 每个 user-candidate pair 的 -d_P score (越大越相似)
    """
    B, L_hist = history_input_ids.shape
    K, L_cand = candidate_sid_seqs.shape[1], candidate_sid_seqs.shape[2]
    # 1. history L0 tokens: 从 history_input_ids 提取 L0 tokens (假设每个 item 用 L0/L1/L2 三层表示, 但我们只取 L0)
    #    更简单: 直接对 history 中所有 token id 用 extract_l0_token, 跳过非 L0 (返回 -1)
    history_l0 = torch.full((B, L_hist), -1, dtype=torch.long, device=device)
    for b in range(B):
        for l in range(L_hist):
            tid = history_input_ids[b, l].item()
            history_l0[b, l] = extract_l0_token(tid)
    # mask: L0 token 有效 (>= 0)
    valid_mask = history_l0 >= 0  # (B, L_hist)
    history_l0_safe = history_l0.clamp(min=0)  # (B, L_hist)
    # 2. history_emb = mean over valid L0 tokens of exp_map_0(L0[token])
    # codebook_t 是切空间 (K_l, d), 映到 Poincaré 球
    codebook_ball = exp_map_0(codebook_t.to(device), c_t.to(device))  # (K_l, d)
    history_emb_ball = codebook_ball[history_l0_safe]  # (B, L_hist, d)
    # mean over valid tokens
    valid_f = valid_mask.float().unsqueeze(-1)  # (B, L_hist, 1)
    history_sum = (history_emb_ball * valid_f).sum(dim=1)  # (B, d)
    history_n = valid_f.sum(dim=1).clamp_min(1.0)  # (B, 1)
    history_centroid = history_sum / history_n  # (B, d), ||·|| < 1
    # 3. candidate_emb: 从 candidate_sid_seqs 提取 L0 (通常第一个 token)
    cand_l0 = candidate_sid_seqs[:, :, 0].clamp(0, 63)  # (B, K)
    cand_emb_ball = codebook_ball[cand_l0]  # (B, K, d)
    # 4. R_geo = -d_P(history_centroid, cand_emb) per pair
    # history_centroid: (B, d), cand_emb_ball: (B, K, d)
    # 输出: (B, K) per pair
    h_exp = history_centroid.unsqueeze(1)  # (B, 1, d) 作为单点 query
    d = poincare_distance_sq(h_exp, cand_emb_ball, c_t.to(device))  # (B, 1, K)
    d = d.squeeze(1)  # (B, K)
    return -d  # R_geo = -d_P (越大越相似, 与 score_orig 同向)


def rerank_with_poincare(orig_scores, history_input_ids, candidate_sid_seqs,
                          codebook_t, c_t, alpha):
    """用 Poincaré distance 重排 candidates.

    Args:
        orig_scores: (B, K) float, beam search 输出的对数概率 (越大越相似)
        history_input_ids: (B, L_hist) long, 用户历史 token ids
        candidate_sid_seqs: (B, K, L_cand) long, 候选 SID 序列
        codebook_t: (K_l, d_tangent) torch, L0 codebook
        c_t: scalar torch, c 值
        alpha: float, R_geo 权重

    Returns:
        rerank_scores: (B, K) float, 重排后分数
        top_indices: (B, K) long, 重排后排序 (top-0 = best)
    """
    R_geo = compute_geo_score_batch(history_input_ids, candidate_sid_seqs, codebook_t, c_t,
                                     device=orig_scores.device)
    # R_geo 量级匹配 orig_scores: orig 是 log-prob, 典型 ~-5 到 0
    # R_geo 是 -d_P, 典型 ~-1 到 0
    # alpha 控制权重
    rerank_scores = orig_scores + alpha * R_geo
    top_indices = rerank_scores.argsort(dim=-1, descending=True)  # (B, K)
    return rerank_scores, top_indices