## 节点 1：冻结 26 条清单；§1 的球面错误已复现并确认

- 正文快照 `results/research/issue244/ISSUE_BODY.md`，sha256 `3f35cacbe0661ca6f9dc8ca59be237e486b069fd8a1aeff1dfe68c2750da15f6`
- 清单 26 条见 `results/research/issue244/RUN_LOG.md`
- 单 Agent，按最新 skill：三状态（本轮若未跑完即 IN_PROGRESS + `PROGRESS.json`，保持 OPEN）

### 先复现 issue 指出的错误（用独立 3D 几何）

单位球面、球极坐标 `x=0.9, y=-0.9`：

| 量 | 值 |
|---|---:|
| 逆球极投影到 `S^2` 后的测地距离 | **2.93126** |
| `4·atan(0.9)` | **2.93126** ✓ |
| 我的 `manifold.dist`（Möbius + `2·tan_k⁻¹`） | **1.56526 ≈ π/2** ✗ |

结论：`Möbius + 2·tan_k⁻¹` 是 **Poincaré 球**的公式；正曲率的球极投影坐标域不是双曲球半径域，
距离必须用球面闭式（逆球极投影后的夹角）。这正是 #243 里 `π/2` 常数与门控塌缩的上游原因之一。

下一步：改 `reference.py` 与 `manifold.py`（k>0 换公式、proju 不再作双曲截断、exp/log 自洽、
k→0 尺度统一），补已知答案与变异测试，再改生产 ego/pairs/评估。

*（`results/research/issue244/COMMENT_01.md`）*
