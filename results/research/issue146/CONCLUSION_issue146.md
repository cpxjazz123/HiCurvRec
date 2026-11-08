# Issue #146 — 结论（几何交付，训练前阻断）

**状态：`RESOURCE_BLOCKED` + `IMPLEMENTATION_NEEDS_FIX`；未启动 Stage2，未导出 SID。**

## 1. 几何定义与曲率推导

径向 warped product（每个残差层的 32 维空间）：

$$
ds^2 = dr^2 + f_\alpha(r)^2\,d\Omega^2,\qquad f_\alpha(r)=r(1+\alpha r^2)
$$

径向截面曲率：

$$
K_{\rm rad}(r)=-\frac{f_\alpha''(r)}{f_\alpha(r)}
=-\frac{6\alpha r}{r(1+\alpha r^2)}
=-\frac{6\alpha}{1+\alpha r^2}
$$

- $r\to0$：$f_\alpha(r)/r\to1$，度量在原点退化为欧氏；$K_{\rm rad}(0)=-6\alpha$。
- $\alpha=0$：$f_0(r)=r$，平面极坐标度量，$K_{\rm rad}=0$。
- 尺度规则（只看 embedding，不读行为图）：$\alpha_\ell = 1/r50_\ell^2$，$r50_\ell$ 为 TRAIN embedding 在该层进入量化前残差范数的 50% 分位数，用初始化后的 encoder 与 KMeans 码本一次性测量并冻结。

## 2. 精确 geodesic 与验证

可旋转对称性把测地线限制在两个径向方向张成的平面内；用弧长参数化求测地线 ODE

$$
r''=\frac{f'}{f}(\varphi')^2,\qquad
\varphi''=-2\frac{f'}{f}r'\varphi'
$$

从 $(r_1,0)$ 积分到极角 $\theta$，对初始角 $\beta$ 打靶使末端半径为 $r_2$；取所有根中最短的弧长。已实现于 `model/warped_rq.py`，并通过以下独立检查：

| 检查 | 结果 |
|---|---|
| $\alpha=0$ 退化为平面欧氏距离 | $(0.5,0.8,0.4)$：0.39134538 vs 0.39134538；$(0.3,0.3,0.7)$：0.20573868 vs 0.20573868；$(1.0,0.6,1.2)$：0.96185794 vs 0.96185794 |
| 同一射线 $\theta=0$ | $d(0.4,1.3)=\lvert 0.4-1.3\rvert=0.9$（精确） |
| 对称性 | $d(r_1,r_2,\theta)=d(r_2,r_1,\theta)$ |
| 曲率公式 | 与 $-f''/f$ 在 $\alpha\in\{0.25,1,4\}$、$r\in\{0.1,0.5,1,2\}$ 上逐点一致（12 位） |
| 小半径极限 | $f_\alpha(r)/r\to1$ |

## 3. 阻断依据（实测，不是估计）

- 当前精确求解（96 点打靶扫描 + `solve_ivp`，rtol $10^{-9}$）实测约 **11 秒/对**。
- 一个训练 step 需要 $4096\times256\times3=3{,}145{,}728$ 对 → 单 step 约 $9.6\times10^{6}$ 秒；按 24 step/epoch、3000 epoch 外推约 **$6.8\times10^{8}$ 小时**。
- 可微性：测地线长度对端点 $(r_1,r_2,\theta)$ 的导数需要打靶根的隐函数定理，**未实现**；现有实现是纯 `numpy/scipy`，没有 autograd 通路。

按 issue 的约束，不允许把“欧氏距离乘半径权重”之类未声明的近似冒充可变曲率，也不允许在成本超预算时换近似继续。因此本候选停在训练前，**没有 SID、没有 D_AF3/retention/Brier 等结构数字**，这些一律记 `NOT_APPLICABLE`（不是 0，也不是 PASS）。

## 4. 要解除阻断需要什么

1. 为打靶根实现隐函数导数（或一个可微的 ODE 伴随求解）并验证梯度方向；
2. 把每对成本降到训练可用量级（闭式或可微近似必须单独声明并重新做数学冻结，不能直接冒充本几何）；
3. 通过四套测试与机器准入后才允许训练唯一候选。

## 5. 产物

- 几何实现：`stage2_RQ-VAE/TIGER_radial_warped_product/model/warped_rq.py`
- 数学规格与参考契约：同树 `math_spec.json`、`reference_contract.json`（连续面为 $K_{\rm rad}$）
- 测试：`tests/test_math.py`（7 项：曲率、极限、$\alpha$ 规则、平面极限、同射线、对称性、成本下界）
- 测试执行证据：`results/research/issue146/suites_TIGER_radial_warped_product.json`
