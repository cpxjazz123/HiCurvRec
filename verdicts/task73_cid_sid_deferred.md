# Task #73 — CID/SID Baselines DEFERRED Decision Log

> **Date**: 2026-07-22 14:21
> **Decider**: AI 自主决策 (per R11.2)
> **Scope**: 子任务 E — SID + CID (P5 framework)

## 决策

**DEFERRING CID/SID baselines** to focus GPU time on the other 4 baselines (ETEGRec + TIGER + TIGER-SAS + LETTER).

Decision threshold is **≥4/6 R@10 ≥ paper - 6%**. With ETEGRec + TIGER + TIGER-SAS + LETTER (4 baselines), we can meet the threshold even without CID/SID.

## 根因 (为什么 CID/SID 跑不通)

LLM-RecSys-ID 上游仓库 (`github.com/Wenyueh/LLM-RecSys-ID`) 写于 transformers 4.26.0 时代，当前环境装的 transformers==5.14.1：

1. **`utils.py`**: `from transformers import AdamW` — removed in 5.x. ✅ 已 patch 用 `torch.optim.AdamW`.
2. **`modeling_p5.py:29`**: `from transformers.modeling_utils import find_pruneable_heads_and_indices` — removed. ✅ 已 patch (未调用).
3. **`modeling_p5.py:35`**: `from transformers import BeamScorer, BeamSearchScorer` — removed. ✅ 已 patch (未调用).
4. **`modeling_p5.py:248`**: `T5Stack(decoder_config, self.shared)` — 5.x T5Stack 签名变化 (只接受 1 个 positional arg + kwargs). ❌ 未解决.
5. **`main.py` line 826-842**: `if __name__ == "__main__"` 块被截断（truncated clone）, 缺 `else: main_worker(0, args, logger)` 分支. ✅ 已 appended.
6. **`CF_indices/` 子目录**: 上游代码查找 `data/{task}/CF_indices/c20_500_CF_index.json`, 我们最初放在 `data/{task}/`. ✅ 已 mkdir + cp.
7. **`item_rep_method.py`**: `change_base()` / `item_resolution()` 函数缺失 (upstream bug). ✅ 已 implement.
8. **`main.py`**: `args.task == "instruments"` 不在 `number_of_items` 映射. ✅ 已 add (24588).

剩余障碍: **T5Stack 签名不兼容** + modeling_p5 整个类设计基于 transformers 4.x 私有 API. 

## 备选方案考量

| 选项 | 时间 | 风险 | 决策 |
|------|------|------|------|
| **继续 patch modeling_p5** | 1-2 h | 高 (剩余多个潜在 api 变化) | ❌ |
| 重建 SID/CID trainer (从零写 T5 seq2seq) | 2-3 h | 中 | ❌ ROI 不及 LETTER |
| **跳过 CID/SID, 集中 4 baselines** | 0 | 0 | ✅ 选定 |

## 数据准备产物 (保留)

所有 CID/SID 训练所需的中间产物已落盘, 未来可直接复用:
- `external/LLM-RecSys-ID/data/instruments/c20_500_CF_index.json` (24556 entries)
- `external/LLM-RecSys-ID/data/instruments/CF_indices/{c20_500, computed_20_500, computed_no_repetition_20_500, computed_optimal_500}_CF_index.json`
- `external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy` (24588 items, 75 MB)
- `external/LLM-RecSys-ID/data/instruments/remapped_sequential_data.txt`
- `external/LLM-RecSys-ID/data/instruments/sentence-t5-id-map.json`

## 上游修复记录 (供未来维护者)

已 patch 的文件列表:
- `external/LLM-RecSys-ID/utils.py` — AdamW 重新导入
- `external/LLM-RecSys-ID/modeling_p5.py` — 移除 find_pruneable_heads_and_indices / BeamScorer / BeamSearchScorer 导入
- `external/LLM-RecSys-ID/main.py` — 补全被截断的 __main__ else 分支 + instruments task 映射
- `external/LLM-RecSys-ID/item_rep_method.py` — 新增 change_base() 与 item_resolution() 函数
- `external/LLM-RecSys-ID/data/instruments/CF_indices/` 子目录补全
