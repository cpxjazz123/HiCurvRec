## 结项:Encoder-only 与码本受控都挡不住坍缩;换监督来源也一样

按本 Issue 的五臂设计跑完(同一 k-means 初始化、300 步上限、权重由逐组上限规则给出)。
SID 一律 `model.eval()` + 正式 `get_indices`,同一 checkpoint 重编码逐项一致(全部 `consistent=true`),
碰撞按同一 21,746 商品集合计算。

| 臂 | 结构 Loss 更新谁 | 实际步数 | 三级碰撞 | L3 唯一码 | 边 AUC |
|---|---|---:|---:|---:|---:|
| `A_none` | — | 300 | **0.0427** | 20,818 | **0.5816** |
| `B_role_teacher_enc` | **只 Encoder** | **50(停)** | 0.8462 | 3,345 | 0.5134 |
| `C_role_teacher_enc_cb` | Encoder+码本,逐层上限 | **50(停)** | 0.7262 | 5,953 | 0.5309 |
| `D_role_teacher_shuffled` | 只 Encoder,目标打乱 | **50(停)** | 0.8219 | 3,872 | 0.5164 |
| `E_role_labels_enc` | 只 Encoder,标签直方图目标 | **50(停)** | 0.9130 | 1,891 | 0.5148 |

梯度实测(第 50 步,未加权):encoder 0.29–0.37;B/D/E 三层码本严格 **0.0**,证明 encoder-only 真的生效。
`collapse_stop.json` 记录四臂在第 50 步触发相对阈值(控制臂同一步 0.292,+0.10)。

**结论:**

1. **Encoder-only 不解决问题** —— 码本辅助梯度为 0 时碰撞仍从 0.043 升到 0.846,是编码器把商品挤进少数码字。
2. **打乱目标同样坍缩**(0.822 vs 真实目标 0.846),所以坍缩来自"q 上的额外梯度"本身,
   不能归因于结构信号;此前"结构目标可学"的数字不能当作结构在起作用。
3. 相对阈值在第 50 步就触发:**训练期门必须默认开启**,不能等 300 步后再看。

**本 Issue 的 Done 不成立**:结构监督误差改善有限,导出 SID 明显坍缩,结构保真(边 AUC)反而低于对照
(0.51 vs 0.58)。按 §19g 不进入完整 Stage2;在解决"任何辅助梯度都摧毁码本划分"之前,这条蒸馏路线不可用。

同轮另有一项与本 Issue 相关的独立结果(详见报告 §11):重设计的**整图一次编码 Teacher** 在 held-out
掩码边上 AUC 0.5123、方向 0.5175、Orbit KL 3.479,仍不如直接用商品 Embedding 的余弦(0.6199)——
即"换一个更好的 Teacher"在当前信息条件下也没有额外收益。

证据:`collapse_gate.py/json`、`collapse_stop.json`、`graph_teacher.py`、`graph_teacher_eval.py`、
`graph_teacher_report.json`、`REPORT_RELATIONAL_DISTILL.md` §10–§11;提交 `fba03b49`,`MANIFEST.json` 逐文件 sha256。
