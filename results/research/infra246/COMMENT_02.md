## 节点 2:先红后绿 —— 已定位并修复的缺陷

### 红(先写测试,现行生产代码在反例上失败)
新增 `tools/graph_infra/tests/test_v2_*.py` 共 39 条断言,首跑 **36 FAIL / 2 PASS**,逐条对应 issue 的 P0 反例:

| P0 | 反例 | 红时现象 |
|---|---|---|
| C5 | 可见 `(7→8)` 权重 3、评估 `(8→7)` 存在 | v1 的 `eval − eval.multiply(visible)` 在 `(7,8)` 得 −2,再对称求和把真实留出方向抵消 → blocked 丢失 |
| C10 | 根有 12 个后继(>K1_OUT=8) | `ego_induced` 用**另一次随机采样**的 out 邻居判角色,超出 8 个的后继被误判为 in |
| C12 | 采样遍历深度 2 但真实 hop 为 1 | v1 的 `role` 存的是遍历深度,不是最短 hop |
| C17 | 三个后继中两个相连、第三个不相连 | v1 要求**整个后继邻域两两独立**,fork 漏判 |
| C24 | `E=[0.3,0.4,0.8]`, threshold 0.5, minimize | v1 无 direction 参数,pass 用 `>=`,worst 取 min,p90 取中心均值 |
| C26 | 前三码 vs 完整含消歧码 | v1 只有单一 collision 口径,SID 长度/取值/缺失/重复无校验 |
| C28 | join 的重复/额外/非有限/图版本 | v1 只校验 shape 与缺失,重复与额外不报错,也不比图版本 |
| C23 | κ→0、近对跖、梯度 | 见下"几何口径" |

### 绿(修复后)
- **隔离**:`isolation.held_out_support` 改为**布尔支撑集差**再对称并集;已知答案 `(8,7)` 必被 block、`(7,8)` 不被 block,v1 抵消公式在同例上得 ≤0(测试同时钉住)。
- **ego**:`topology.root_roles` 读真实 CSR;`hop` 按根的真实邻接重算,`traversal_depth` 另存;`root_local` 显式;`cap` 明确定义为**邻居数**(根另计);新增 `ego_full`(确定性完整一跳)、`batch_egos` 补齐 `edges/edge_mask/features`。
- **graphlet**:新增 3 节点 canonical(6 置换取最小)、自同构轨道、16 个一一对应的族名、`fork_relaxed/merge_relaxed` 存在性定义、`motif_instances` 实例枚举、`family_counts` 全量向量化计数(42.2M 邻接对全覆盖)、4 节点抽样接口。
- **指标**:`subgraph_summary` 显式 `direction`;失真口径 pass=`E<=t`、worst=max、p90=中心内 quantile(0.9) 后汇总;新增 `sid_structure` / `sid_audit` / `codebook_stats(unit=)`。
- **join / manifest**:新增 `entry_v2`(仓库相对路径、`legal_use`、`graph_version`、源码与数据哈希)、`audit_manifest`、`assert_legal_use`、`join_scores_v2`(shape/dtype/重复/缺失/额外/非有限/图版本全查,按 pair id 对齐,位置对齐从不使用);`verify_manifest` 只校验不写。
- **PPR**:新增 `ppr.solve_selected`,用稀疏 LU 精确求解并把悬挂节点质量归还源点,残差逐源上报。

### 几何口径更正(会影响后续臂,必须记录)
v1 的 κ-立体算子在球面分支与双曲分支用了**不同的 2 倍约定**(球面 `d(0,exp(v))=|v|`,双曲 `=2|v|`),且 `k=0` 返回普通欧氏范数,与曲率族相差 2 倍。v2 统一为:`d(0, expmap0(v)) = 2|v|`(所有 k)、`lim_{k→0} d_k = 2|x−y|`,球面 exp/log 同步改为 `R tan(|v|/R)` / `R atan(|p|/R)`。**k≠0 的距离数值不变**,变的是球面 exp/log 与 k=0 分支;`test_v2_geometry.py` 与 `test_known_answers.py` 同时钉住 `2|v|` 与连续极限。

### 变异测试
`tools/graph_infra/tests/test_mutation_v2.py`:14 个变异(数值 5 / 结构 6 / 路由 3),判定为"未变异副本先通过 + 变异副本出现指定 `FAIL` 行,崩溃不计",**14/14 全部被抓到**(`results/research/infra246/MUTATIONS_V2.md`)。其中 `held-out-uses-the-v1-weighted-formula` 由 C5 的已知答案抓住,`ego-role-from-a-truncated-sample` 与 `ego-hop-equals-traversal-depth` 由 ego 反例抓住,`fork-back-to-universal-independence` 由三后继反例抓住,`manifold-flat-limit-reverts-to-plain-norm` 与 `sphere-exponential-uses-the-half-angle` 由几何边界抓住。

下一步:v2 资产全量冻结 + 隔离审计 + TIGER 来源校验,然后结论对账。
