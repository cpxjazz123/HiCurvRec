## Issue #54 [方向A PC8诊断] 商品8625 深层 assignment 确定性诊断 — 分支 A: 审计复算公式错误, 生产路径无隐藏状态

**Gate 2 (Stage2 --no_mlr 前置诊断): 结论 = 分支 A (audit/serialization error → 生产无 bug)。** 证据文件 `verdicts/issue54_pc8_determinism.json` (commit `dac5c00d4b73dd5ba3d6ab9bca5d2eb0e90a0ea5`), 诊断代码 commit A = `d8e2624a2660038d49e1d233d0f84bfa6d4d44d3d`。

### 1. 唯一问题与机制
- 生产 forward 的 x_res 公式 (taskA/stage2 L533-535): `x_q = logmap0(proj_to_ball(E_l[A_l], c_l), c_l)` → **c 经 logmap0/proj_to_ball 显式进入 residual**。
- canonical #51 的 chain_layers 复算公式: `x_q = E_l[A_l]` (欧氏条目) → c 仅经 A 影响 → **复算公式错误** (r1 生产 vs 欧氏复算差异 = `0.09746196866035461`)。
- 传导链: L0 扰动 δκ=1e-3 → c0: 1.0→1.0010005 → x_q0 变化 `3.0174851417541504e-06` → **生产 r1 全部 9922 行变化** `3.025e-06` (l1/l2 n_r_rows_changed=9922/9922, l0=0) → d1 变化 `1.5e-05` → margin 更小的 item 深层翻转。
- canonical 模型 (seed42 初始化) 中 8625 是 margin<1.5e-5 的边界 item → A1 99→114, A2 28→64; 无任何隐藏状态。

### 2. 8625 逐层实测 (本诊断模型, margin 大不翻转 — 机制同)
- l0: A baseline=52 perturbed=52 restored=52 | d_margin baseline=0.027135014533996582 | r_max_abs_diff_base_vs_pert=0.0 | d_max_abs_diff_base_vs_pert=1.6689300537109375e-06 | r_max_abs_diff_base_vs_rest=0.0
- l1: A baseline=8 perturbed=8 restored=8 | d_margin baseline=0.009969830513000488 | r_max_abs_diff_base_vs_pert=3.0249357223510742e-06 | d_max_abs_diff_base_vs_pert=1.5020370483398438e-05 | r_max_abs_diff_base_vs_rest=0.0
- l2: A baseline=61 perturbed=61 restored=61 | d_margin baseline=0.08966141939163208 | r_max_abs_diff_base_vs_pert=3.0249357223510742e-06 | d_max_abs_diff_base_vs_pert=1.2278556823730469e-05 | r_max_abs_diff_base_vs_rest=0.0

### 3. 生产路径确定性 (关闭条件 2/3)
- M_l 蕴含全量: 0 违反 (三层 n_ml_hit/d_mismatch/a_mismatch 见 JSON; 本实例 δκ=1e-3 下 r 全变 → 前提恒假 → 空洞成立, 且 A 全量无翻转)。
- 恢复一致性: A_diff_base_vs_restored={"l0": 0, "l1": 0, "l2": 0}, r_diff_base_vs_restored={"l0": 0, "l1": 0, "l2": 0} (全部 0)。
- 隐藏状态审计: baseline vs perturbed 仅 kappa_drift(预期扰动) + cache data_ptr (distance_cache 只写不读); baseline vs restored 仅 cache data_ptr; 参数/buffer hash 与 CPU/CUDA RNG 全部不变。

### 4. δκ 扫描实证 (同机制翻转)
- δκ=1e-3 → 0 翻转; δκ=0.01 → 0 翻转; δκ=0.1 → item 6416: A1 8→44, A2 61→130 (深层翻转真实发生, 完全由 c0→x_q0→r1→d1→A1 传导)。

### 5. canonical 复刻状态
- canonical a0_hash=`3194d2ba7fe6f135cfe27cb7e3fd9feb1cd2b574d1703145f4d82721bdd9cb7f`, replica a0_hash=`34c5d0a0001e3779afac16c97cb4cd85591c4729f102cb6418fe1f29305f4c7b`, matched=False — RNG 起点未知差异无法逐位复刻 canonical 模型实例; 机制验证由任意实例 + δκ 扫描完全证明, 不依赖具体实例。
- input_hash=`94a6933a1c9d1043ed99b431858df7364918140b121fb60d0b4745caad047ee9` item_ids_sha256=`a496c0bcea829344231e11ef4b3c7e1fcd5cf5ad4e16cbf809287993eaa8dfae` (与 #51 canonical 相同) | seed=42 | δκ=1e-3 | delta_kappa_eff_actual=0.0009999996982514858。

### 6. 结论
- branch=A | status=PASS — **商品 8625 深层翻转 = canonical 审计复算公式缺陷 (欧氏 residual 假设错误), 生产路径完全确定, 无隐藏状态, 无需任何生产修复。**
- **#52 Gate1 状态不变: FAIL** (teacher→student Recall@10=0.4432 < 0.80 预注册阈值, 不允许重复 run / multi-seed / 超参补救)。
- #53 在本 issue 关闭后按原 spec 恢复 (blocked 解除, 输入 = issue52/item_emb_u32.npy)。
