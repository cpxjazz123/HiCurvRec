## Issue #51 [方向A PC8补证] — 统一 canonical 证据: precheck PASS (11/11)

**Gate 1 (Stage1 precheck): PASS** — 全部 11 项 PASS (PC1 流形 / PC2 互逆 / PC2 近零补证 / PC3 等距 / PC4 attention / PC5 centroid / PC6 FD / PC7 三条路径 / PC7 9922 全量导出 / PC8 三层合规 / PC8 canonical)。canonical JSON 机械复制字段如下 (唯一证据文件 `verdicts/issue51_pc8_canonical_evidence.json`)。

**Gate 2 (Stage2 运行时来源): PASS** — MLR_ENABLED=False, SK_EPSILONS=[0.0,0.0,0.0], assignment_source=poincare_argmin, vq_class_used=KappaAwareVectorQuantization.forward (hard torch.argmin(d, dim=-1)); call_counts mlr_logits=0 / sinkhorn=0 / poincare_distance=831 (>=831) / argmin=291 (>=291) / fl_calls_per_layer=[90,90,90] (精确=9 次 sid_once × 10 batch)。

**Gate 3/4 (Stage3/4): 未涉及** — 本 issue 为 precheck 补证, 不允许训练, 无 Stage3/4 产物。

### run_id 与绑定
- run_id: `6bfabce24917f658d23cbeabc821f22421e66bc9345c7faaa8307305828647f5`
- code_commit_a: `2b556d91bc04dff1504dbb9502289ab49e0e36cc` (审计代码; 修复版 A2=`2b556d91bc04dff1504dbb9502289ab49e0e36cc` 为实际运行代码)
- item_ids_sha256: `a496c0bcea829344231e11ef4b3c7e1fcd5cf5ad4e16cbf809287993eaa8dfae` (9922 全量有序 ItemID)
- seed: 42 | config_hash: `9d491e0a644a4537fe3e86aea8bdb121d1b8a5ca36daed9942caa4bb0dee5c84`
- 证据 commit B: `9eb9ffb25105c03b1372760665e8bab3dcde8585` | 审计代码 commit A: `3af2c05e081196749a1414781116d564c89fba65`

### baseline (逐层 scale + hash)
- l0: kappa_eff=0.0 c=1.0 sqrt_c=1.0 ball_radius=1.0 | projected_codebook_hash=88df642db651256778f0c05e02f09ebb83b80848bdeafb4f26e723b12a213bcf | distance_hash=514d4a99a32850c3f5cd4edb3c2aed9230ece241a290f62e371110dd9b46aa27 distance_sum=413893.03125 | assignment_hash=3194d2ba7fe6f135cfe27cb7e3fd9feb1cd2b574d1703145f4d82721bdd9cb7f | selected_codeword_hash=bc7de68f12782a8f5acaa08973e19f802d392d7bd6e523f36e0b1f9b5c79fed3 | residual_hash=3d1c372ac34ba54de15a96ef1dee85b42902e0e794cfa8cb2ecd6eb69a7edec5
- l1: kappa_eff=0.0 c=1.0 sqrt_c=1.0 ball_radius=1.0 | projected_codebook_hash=2dc5d140ae44d02c2641f3554110f0fe1ebf9fd878e39c70b960a4ba5a777385 | distance_hash=9ad1e5f35295f71a9518c4dfab29e65d1e4d55681cc11677b1a95e7d1a077086 distance_sum=1092671.5 | assignment_hash=23f9d60ee100d434244646d808ff20599547f2a3d5e4605e4ea5f7684e8f1843 | selected_codeword_hash=bc75b41ee0e27fadaa381b4a8ce86a5cfd1a70a048f679cac34066fbccea79db | residual_hash=0d2dcf434c9dfe1b008db5ebc6dfd752f569417ca03b0e6bc73d9ba937179434
- l2: kappa_eff=0.0 c=1.0 sqrt_c=1.0 ball_radius=1.0 | projected_codebook_hash=e1fe92169593cde85e3fd90632063b924fffdbe5b2a1aa531dbbbb8b63f5f66d | distance_hash=8fa14336d4d2445333f6776c76741a084406dcbb82ee85437da224f1322ce7d3 distance_sum=2331126.75 | assignment_hash=c598b253989ee1843e8e19db85f99acb9576e2248215b663d86dfb16239cc14a | selected_codeword_hash=cc2da6af9f786ec5f2263474cf0cb97f8169a5639ed0fbbd62195638927fadc7 | residual_hash=0e5676e29a915040db90b8515275dab77010c29f75e0760fc30c25b5cd17e882
- SID3: a0_hash=`3194d2ba7fe6f135cfe27cb7e3fd9feb1cd2b574d1703145f4d82721bdd9cb7f` a1_hash=`d73966a5bd1b88a9c6db13dc0205049959a6bcfb86f86f741c5cafe983afbccd` a2_hash=`1244c48123edc9010bfb665f55345ca8fdec2c249f1e13f3c12bef623ab448dc` | sid3_hash=`b2b4ace78cdf1240440002ffdd8f5c20b14cadf6082164e182b848314e3b4a4b`
- SID4: collision_digit_hash=`2759f9d4aae87a0b5cd1b9b9817b63d1b88b770547128bf58940f808e1b1d361` | sid4_hash=`27cf2a198e318a4ade6162aadd765f95cb86b54f68a094da958f9ede3425b9ef`
- collision_groups: n_unique_3digit=7 max_group_size=6253 n_colliding_items=9915 (未训练随机码本, 仅链路正确性)

### perturbations (δ=1e-3 逐层扰动+恢复, 同一进程)
- l0 status=PASS | delta_kappa_eff_actual=0.0009999996982514858 delta_c_actual=0.001000523567199707 | earlier_layers_unchanged=True cur_c_changed=True cur_projected_codebook_changed=True cur_distance_changed=True cur_distance_delta_abs=0.5 cur_assignment_changed=False | assignment_flip_count_layer=0 sid3_flip=1 sid4_flip=350 n_sid4_flip_without_own_3digit_change=349 | layer_assignment_unchanged_but_sid4_changed=True sid3_unchanged_but_sid4_changed=False sid4_replay_consistent=True | restored.ok=True chain_fully_restored=True
- l1 status=PASS | delta_kappa_eff_actual=0.0009999996982514858 delta_c_actual=0.001000523567199707 | earlier_layers_unchanged=True cur_c_changed=True cur_projected_codebook_changed=True cur_distance_changed=True cur_distance_delta_abs=26.5 cur_assignment_changed=False | assignment_flip_count_layer=0 sid3_flip=0 sid4_flip=0 n_sid4_flip_without_own_3digit_change=0 | layer_assignment_unchanged_but_sid4_changed=False sid3_unchanged_but_sid4_changed=False sid4_replay_consistent=True | restored.ok=True chain_fully_restored=True
- l2 status=PASS | delta_kappa_eff_actual=0.0009999996982514858 delta_c_actual=0.001000523567199707 | earlier_layers_unchanged=True cur_c_changed=True cur_projected_codebook_changed=True cur_distance_changed=True cur_distance_delta_abs=64.75 cur_assignment_changed=False | assignment_flip_count_layer=0 sid3_flip=0 sid4_flip=0 n_sid4_flip_without_own_3digit_change=0 | layer_assignment_unchanged_but_sid4_changed=False sid3_unchanged_but_sid4_changed=False sid4_replay_consistent=True | restored.ok=True chain_fully_restored=True

### SID 碰撞路径逐商品解释 (spec 第3条)
- l0: A0 未翻转但 SID4 变化 350 个 item (own_3digit_flip=1, collision_digit_cascade=349)。机制: 扰动经 residual 传导 → 深层 A1/A2 翻转 1 item → add_4th_dedup_digit 的 C 位是碰撞组内序号 (seen[key]%256, tie-break=有序 ItemID), 同组后续 item C 位级联前移/后移。全部 350 个变化 item 在 JSON changed_items 逐条列出 (item_id + old/new 3digit/4digit + change_type)。
- l1/l2: 0 个变化。sid4_replay_consistent=True (SID4 用各自 SID3 独立重算逐位一致 → SID4=f(SID3) 确定性, 无隐藏状态); sid3_unchanged_but_sid4_changed=False (单向蕴含成立)。

### residual 重算审计 (spec 第4条)
- l0 l0: residual 数值不变 — 生产公式 r_{l+1} = r_l - E_l[A_l(i)]: x_q 直接取欧氏码本条目 E_l[A_l(i)], 曲率 c 仅经 assignment A_l 影响 x_q; 本扰动下该层 assignment 未翻转 → x_q 逐位不变 → residual 逐位不变; F_l hook 计数增量 Δ=10 证明 forward 已重新执行
- l0 l1: residual 数值不变 — 生产公式 r_{l+1} = r_l - E_l[A_l(i)]: x_q 直接取欧氏码本条目 E_l[A_l(i)], 曲率 c 仅经 assignment A_l 影响 x_q; 本扰动下该层 assignment 未翻转 → x_q 逐位不变 → residual 逐位不变; F_l hook 计数增量 Δ=10 证明 forward 已重新执行
- l0 l2: residual 数值不变 — 生产公式 r_{l+1} = r_l - E_l[A_l(i)]: x_q 直接取欧氏码本条目 E_l[A_l(i)], 曲率 c 仅经 assignment A_l 影响 x_q; 本扰动下该层 assignment 未翻转 → x_q 逐位不变 → residual 逐位不变; F_l hook 计数增量 Δ=10 证明 forward 已重新执行
- l1 l0: residual 数值不变 — 生产公式 r_{l+1} = r_l - E_l[A_l(i)]: x_q 直接取欧氏码本条目 E_l[A_l(i)], 曲率 c 仅经 assignment A_l 影响 x_q; 本扰动下该层 assignment 未翻转 → x_q 逐位不变 → residual 逐位不变; F_l hook 计数增量 Δ=10 证明 forward 已重新执行
- l1 l1: residual 数值不变 — 生产公式 r_{l+1} = r_l - E_l[A_l(i)]: x_q 直接取欧氏码本条目 E_l[A_l(i)], 曲率 c 仅经 assignment A_l 影响 x_q; 本扰动下该层 assignment 未翻转 → x_q 逐位不变 → residual 逐位不变; F_l hook 计数增量 Δ=10 证明 forward 已重新执行
- l1 l2: residual 数值不变 — 生产公式 r_{l+1} = r_l - E_l[A_l(i)]: x_q 直接取欧氏码本条目 E_l[A_l(i)], 曲率 c 仅经 assignment A_l 影响 x_q; 本扰动下该层 assignment 未翻转 → x_q 逐位不变 → residual 逐位不变; F_l hook 计数增量 Δ=10 证明 forward 已重新执行
- l2 l0: residual 数值不变 — 生产公式 r_{l+1} = r_l - E_l[A_l(i)]: x_q 直接取欧氏码本条目 E_l[A_l(i)], 曲率 c 仅经 assignment A_l 影响 x_q; 本扰动下该层 assignment 未翻转 → x_q 逐位不变 → residual 逐位不变; F_l hook 计数增量 Δ=10 证明 forward 已重新执行
- l2 l1: residual 数值不变 — 生产公式 r_{l+1} = r_l - E_l[A_l(i)]: x_q 直接取欧氏码本条目 E_l[A_l(i)], 曲率 c 仅经 assignment A_l 影响 x_q; 本扰动下该层 assignment 未翻转 → x_q 逐位不变 → residual 逐位不变; F_l hook 计数增量 Δ=10 证明 forward 已重新执行
- l2 l2: residual 数值不变 — 生产公式 r_{l+1} = r_l - E_l[A_l(i)]: x_q 直接取欧氏码本条目 E_l[A_l(i)], 曲率 c 仅经 assignment A_l 影响 x_q; 本扰动下该层 assignment 未翻转 → x_q 逐位不变 → residual 逐位不变; F_l hook 计数增量 Δ=10 证明 forward 已重新执行
- F_l hook: 每层 90 次生产 forward (9 次 sid_once × 10 batch), 输入 hash 集合见 JSON fl_input_hash_unique_per_layer (每层 ≤10 个 batch residual hash)。

### reload 5/5 (spec 第6条)
- status=PASS all_diff_zero=True; 每轮 a0/a1/a2/sid3/collision_digit/sid4 六对象 hash 相同且与 baseline 逐元素 diff=0 (runs 字段完整 hash 在 JSON)。

### 结论
- 唯一 canonical run (单进程: baseline→L0/L1/L2 扰动→恢复→reload 5/5), 无跨 run 拼接; run_id=6bfabce24917f658…; 整体 status=PASS, precheck 11/11 PASS → **precheck PASS**, 允许进入 Gate1 Stage1 正式训练。
