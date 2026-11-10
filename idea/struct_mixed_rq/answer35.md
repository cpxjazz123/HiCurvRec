# answer35 — Stage3 跑完 150 epoch;`test_recall@10 = 0.054597`,未过 0.065
**状态:完成一次完整 Stage3;低于 0.065,也低于基线第一次记录;不据此判机制成败**

承接 `answer34.md`。先接受 §0 的两处口径更正:
1. `training_hooks.py` 磁盘实际为 `4046b3f8c7680dc7d9b4534b9069347792a5e224ba5a641bedfd8cd38e01fc34`(q34 §7 的 `8ede9c09…` 是旧值);
2. 两个 collision 定义分开写:训练器 `collision=0.1108309…` 是 4-token 全串唯一度;`l3_collision=0.1855452…` 是前三个语义码的 item-weighted 重复占比。以下只用后者的名字报 §63 值。

## 1. 启动与协议
- `stage3_start("results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/item_sids.json")`,pid `3813051`;
- 协议核对:`NUM_EPOCHS=150`、`NO_EVAL=True`、`SKIP_TEST=False`、`SEED=42`、`BEAM_SIZE=20`,`mismatched={}`,`early_stop_present=false`;
- 布局:`ddp4`,`world_size=4`,`train_batch_per_process=4096`,`global_train_batch=16384`,`infer_batch_per_process=1024`,`CUDA_VISIBLE_DEVICES=0,1,2,3`,端口 50201;
- SID:`item_sids.json` sha256 `1839d1cea74dbc0eb5789c88b3fda1d69ff87586c561ea7b1c19ca41a5ec61d3`。

## 2. 完成情况
| 项 | 值 |
|---|---|
| 训练 | 150/150,`cumulative_time_s=1801.103`(≈30.0 min) |
| 最终 test | 2026-11-10T13:26:07+11:00 写出结果 |
| 进程 | 训练与测试结束,pid 处于已终止状态(Zs,已不可取退出码) |
| 结果目录 | `results/stage3_T5Train/TIGER_struct_mixed_rq_ste_fix/` |
| test 文件 | `logs/Amazon_2023_Instruments/Nov-10-2026_12-53-45/test_final.json` |

## 3. 最终指标(test_final.json 原值)
| 指标 | 值 |
|---|---:|
| `n_eval` | **57439** |
| `test_recall@5` | 0.03584672435105068 |
| **`test_recall@10`** | **0.05459705078431031** |
| `test_ndcg@5` | 0.023696594639440724 |
| `test_ndcg@10` | 0.02973556061737681 |

## 4. 两个判断分开写
- **(a) 与合法基线的差**:`TIGER_RQ-VAE` 两次实测
  `0.05553717857205035` 与 `0.04019916781281011`;
  本候选 `0.05459705078431031`,
  相对第一次 **−0.0009401277877400388**,相对第二次 **+0.0143978829715002**。
- **(b) `test_recall@10 > 0.065`**:**不成立**(0.054597 < 0.065)。

不据此判定机制成功或失败;按 §19 保留该候选记录。

## 5. R2 归档(§6b)
`python3 results/research/issue139/upload_stage3.py TIGER_struct_mixed_rq_ste_fix`
→ `r2:genrec/58/`,11 个文件,recall 键 6 个;`r2_ledger.json` 已写入 `58`:

| 文件 | SHA-256 |
|---|---|
| `TIGER_struct_mixed_rq_ste_fix/train_rqvae.py` | `61dd1d9ada2b02aa752b9832951275819186c7e55d7d6781a761589d6160add9` |
| `…/math_spec.json` | `7e1f233790e3de44c9b6a9aeb295abdc2de0dd6e4f06ff4e8ed42e01f4dd8d4d` |
| `…/reference_contract.json` | `6b2a4187f99694343b1012f302995150519b740e481937d062a6f28fab07d31d` |
| `item_sids.json` | `1839d1cea74dbc0eb5789c88b3fda1d69ff87586c561ea7b1c19ca41a5ec61d3` |
| `rqvae_best.pth` | `9da6aa4baa3f12bf992b48d4f06d0d48755945d30d1b8a42335487f0749f0e06` |
| `recall.json` | `994af4ab68f9be43daccd2c08abe8352fac74e887096ec18e8a52bae72060574` |

## 6. 日志与哈希
| 文件 | SHA-256 |
|---|---|
| `…/Nov-10-2026_12-53-45/test_final.json` | `415507a2ff607066c76d2e1ec33750a2153f691f67cb8cb422900c1bae7d49a2` |
| `…/Nov-10-2026_12-53-45/HG_Rec.log` | `cf3ff9362b1a9b091d89438781f074e94323ab279bc5141f6a927ccb352e504d` |
| `…/Nov-10-2026_12-53-45/training_metrics.jsonl` | `d40baf6db5d036b83b8c644b47d30ce6c7028ab169228afe3d8db050234fa2f8` |

## 7. 本轮未做
未改冻结协议、未恢复 early stop、未换种子、未做调参或开关消融、未因 Stage2 诊断值挑选启动。
