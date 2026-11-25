R02 完整跑完并结案为**负结果**（CHECK_10：16 PASS / 1 FAIL / 0 NOT_RUN，唯一 FAIL 是关闭阈值 C17）。

事实链：Stage2 四卡 3000 epoch 用时 2116.782s、final-epoch checkpoint、无早停；最终 checkpoint 精确复现全部 SID（三码碰撞 0.06491235205596457 < 0.4，量化误差 0.007673，有效秩 5.052）；Stage3 按冻结协议 150 epoch、beam20、seed42 在完整 57439 行 test 上评测；独立复算持久化预测行**完全一致**：test_R@10=0.05266456588728913（strict_target_met=false）；完整 57439 行推荐验证 validation_R@10=0.0573129755044482；四件套已归档 r2:genrec/103/（127 文件、23 个 recall 键）；分层指标按 §63 落盘（conditional_g3 明确标 MISSING，未编造）。

**预注册假设被证伪**：每 50 个原生 epoch 重刷一次 maximal global module budget，相对 R01 只刷新一次，test 差 −0.000070、validation 差 −0.000331，两个划分上都是零效应，远低于 0.065 门槛。

**失败点被定位而非猜测**：按 SID 文件配对 14 个臂后，最终 Recall 与 level-1 承诺粒度强相关（绝对 L1/L2/L3 前缀命中率、refinement gain、distinct 二码前缀数，|r| 0.80–0.91）。本家族产出的是近乎逐物品的 SID（8840 个二码前缀、三码碰撞 0.0649、量化误差 0.0077），L1 前缀只带住 6.43% 的验证目标，而最好的混合臂带住 15.99%。

**关于你要求的结论，现有实验给出的是分级答案**（`results/research/issue256/analysis/CONCLUSION_MIXED_CURVATURE_AND_ROLE_BIAS.md`）：
- *混合曲率更适合*——成立。同容量同协议下 E16×H16 混合空间比 E16×E16 纯欧氏高 +0.003055（12 个重复臂的中位噪声 0.000949、最小 0.000052；同一混合臂两次 Stage3 只差 0.000052），全仓库最高分 0.057905 也是混合臂。
- *不同graphlet需要不同几何偏置*——方向一致但幅度未确立：per-role vs 置换角色对照 +0.001114，per-role vs 全局 +0.000157，local vs global α +0.000296，都处在噪声底附近；同时 Stage2 结构差异很大（L1 hard ER 3.85 / 5.26 / 4.51，二码前缀 3421 / 4920 / 4386），说明几何确实跟着 graphlet 角色走，但 Recall 转化量小于协议噪声。

下一轮单点改动已定：把角色条件化的混合曲率距离写进 level-1 分配代价（Sinkhorn 均衡分配与冻结配方不变），目标是绝对 L1 命中率回到 0.10–0.16、二码前缀数回到约 4000，并预注册可证伪预测后再训练。Issue 保持 OPEN。
