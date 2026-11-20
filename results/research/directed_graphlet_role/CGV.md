# CGV6：可执行证据索引（issue #221 §6）

每一类都指向**本机上实际跑过**的测试或日志；`cgv/*.log` 是最后一轮的原始输出。

| # | 要求的检查 | 证据（文件 → 覆盖点） | 最近一次结果 |
|---|---|---|---|
| ① | 有向图统计、唯一构图、TRAIN-only、SHA、反向/互惠边 | `tests/test_directed_graph.py`：同一有序对只存一行且权重可重数、反向边单独保留、无自环且 ID 合法、CSR 非对称、度数等于边表、无向参照覆盖每条有向边、输出与记录的 sha 一致 | 7 passed（含在 `cgv/tests_issue_dir.log` 的 25 项里） |
| ② | subtype/orbit 规范化、置换不变/等变、反向边、稀有类、split 防泄漏 | `tests/test_directed_graphlets.py`：converging vs diverging 楔形不同类、orbit 随有向角色、互惠边不等于普通边、分类对置换不变、split 只看节点集合、池里带的是有向标签、orbit id 不依赖输入顺序、源/汇在反转下稳定、合并 subtype 不混用 `__other__`；`tests/test_label_modes.py`：无向模式 5 类 / 有向 47 类 / 打乱保骨架但改类型；`tests/test_dgdv_leakage.py`：held-out 对双向删除 | 25 passed |
| ③ | L_type、L_role 的独立 loss/梯度路径与方向 | `stage2_RQ-VAE/TIGER_drq_D1/tests/test_directed_arms.py`：D1 只到 encoder、D2 旧 STE 对 codebook 梯度为 0、D3 同时到 encoder 与 codebook、D4 冻结头不动、一步更新确实降 joint loss、orbit mask 只放行该 subtype 的 orbit | 7 passed（`cgv/tests_arm_gradients.log`） |
| ④ | 多卡 DDP 参数同步、head freeze、真实 optimizer step | `tools/research/stage2/tests/test_collectives.py`（rank 分支内的集合通信静态拒绝 + 两 rank gloo 功能守卫）、`tests/test_launch_determinism.py`；head freeze 由 ③ 的 D4 用真实 step 断言 | 13 passed（`cgv/framework_guards.log`） |
| ⑤ | 完整入口、checkpoint、双进程 hard SID 重编码 0 mismatch | `tools/research/stage2/tests/test_export_check.py`（训练器存的表 vs checkpoint 重编码必须一致，故意造不一致必须报错）、`test_stage3_determinism.py`；每个臂的 `export_manifest.json` 记 `item_sids.json` 的 sha | 同上 13 passed |
| ⑥ | 故障注入必须报错 | 集合通信塌缩：`test_collectives` 静态守卫；码本塌缩：D2/D3/D4 的 `assignment_floor.json` 真实触发并停止；标签/泄漏：`test_dgdv_leakage.py`；本轮参考臂的两个缺陷（无向词表缺 `allowed` → KeyError、打乱模式把局部 orbit 当全局 → loss 非有限）由 `stage2_RQ-VAE/TIGER_drq_{U,S}/tests/test_reference_label_modes.py` 复现并锁死 | U 4 passed / S 3 passed（`cgv/tests_U_arm.log`、`cgv/tests_S_arm.log`） |

## 诚实边界

- ③ 的梯度证据来自 **D1 树**的测试；U/S 与 D1 共用同一 `drq.py`/`directed_role.py`，只改 `LABEL_MODE`，所以这两棵树里该文件被显式 skip 并注明理由，而不是放一个更弱的替身。
- ④ 的多卡证据是**框架守卫**（两 rank gloo + 静态拒绝），不是一次 4-rank 有向臂训练：按仓库 §19d0b，单卡主机上的有向臂走单进程分支、全局 batch 仍为 4096。
- ① 的 TRAIN-only 由构图脚本的输入（`train.parquet`）与 `graph.npz` 的 sha 断言承担；本轮没有再跑一次全量构图（图已冻结并通过 sha）。

## 复跑命令

```bash
# ① ②
OMP_NUM_THREADS=4 /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  -m pytest -q results/research/directed_graphlet_role/tests/
# ③
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  -m pytest -q stage2_RQ-VAE/TIGER_drq_D1/tests/
# ④ ⑤ ⑥（框架守卫）
OMP_NUM_THREADS=4 /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m pytest -q \
  tools/research/stage2/tests/test_collectives.py tools/research/stage2/tests/test_export_check.py \
  tools/research/stage2/tests/test_launch_determinism.py tools/research/stage2/tests/test_stage3_determinism.py
# ⑥（参考臂标签模式）
OMP_NUM_THREADS=4 /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m pytest -q \
  stage2_RQ-VAE/TIGER_drq_U/tests/ stage2_RQ-VAE/TIGER_drq_S/tests/   # 分两次跑，模块同名
```
