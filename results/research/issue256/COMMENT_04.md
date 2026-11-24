R01预注册（先于参考/候选实现）：只改初始曲率系数的校准策略。由真实首个global4096训练batch的全局平均VJP向量，对17个曲率耦合模块施加 `lambda*||g_curv,g||<=||g_original,g||`，取满足全部约束的最大正lambda，之后冻结。原始重建/commitment/codebook必须全进分子；先平均向量再求范数，不平均各卡范数。其余三曲率/完整2hop/共享有向GNN/分层gate/累计前缀/实际硬分配和3000/150配方不动。

依据是训练/validation而非test样本：旧真实首步17组x4卡，结构/原始梯度压力最高121.021（Query1）、codebook1最高103.123；原校准仅看256排序商品上的encoder。高L1保持同时出现在度匹配无关对（validation对照0.920694、直接边0.922230），不能称选择性保持。根因仍是假设：尺度漂移/代理噪声/AdamW等替代解释未排除。

训练前预测：全局17组预算逐项满足、至少一组饱和、四卡系数相同、参数不因校准更新；独立SciPy LP/手算/变异及真实入口证明。训练后碰撞须<0.4、三元组数>457；分别报告重建与validation选择性、几何宏微/符号和最终Recall。失败就回R1/R2，不同机制追分重跑。

冻结证据：`rounds/r01/CONTRACT.md` SHA256 `aa644b16eb9a378d48384eb72eb618eac8b90efe4790baad2e2fe7b2f2a54156`；`DIAGNOSIS.md` SHA256 `8c2c2a2ecf7b232bde8e48ad2571e69a7d812ad8d02875cb6ee75aaf641ccadf`；`PARENT_DIAGNOSTICS.json` SHA256 `4f884d6f9bbfbc439d96a914a8de81fe279f172ac2e4ff9e550212a3d50f81b5`。候选TIGER_module_budget_prefix_rq、正式run issue256_r01_formal（尚未派发）。
