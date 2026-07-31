# Task #367 / Issue #74 Gate 1 — sid_metadata 到 attention-bias 合同预检 PASS (R18 强制实证)

**日期**: 2026-07-31
**触发**: Issue #74 [方向C 预检] sid_metadata 到 attention-bias 合同 — GitHub OPEN
**前置**: Issue #71 闭环完成 (task363 Gate 1 precheck blocked, 0/7 markers 缺失)
**任务**: precheck 静态审计 + 实施 sid_metadata + Stage3 batch 序列化 + attention-bias stub + 7 markers 验证
**结果**: ✅ Issue #74 预检 PASS, 7/7 markers 全部 PASS (R18 强制实证)

---

## 1. R17/§19 Gate 决策

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏸ STOP (Issue #74 预检任务) | Issue #74 spec 是预检任务, 不要求 Stage 1 训练 |
| **Gate 1.5 = 预检 7 markers** | ✅ **PASS** | 7/7 markers 全部 PASS |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

## 2. 7 markers 全部 PASS (R18 强制实证)

### 2.1 7 markers 验证 (Issue #74 spec 强制)

| Marker | 内容 | 验证方法 | 结果 |
|------|------|------|------|
| **M1 sid_metadata schema** | layer_id, kappa_l, scale_l, assignment_confidence, mask | dataclass 定义 | ✅ PASS |
| **M2 Stage3 batch 序列化** | metadata_list -> dict | serialize_metadata(metadata_list, device='cuda:0') | ✅ PASS |
| **M3 Stage3 batch 反序列化** | dict -> metadata_list | deserialize_metadata(serialized) | ✅ PASS |
| **M4 attention-bias stub 实现** | 可关闭 nn.Module | AttentionBiasStub(hidden_dim=512) | ✅ PASS |
| **M5 关闭 ↔ vanilla T5 等价** | stub.enabled=False → out == hidden | torch.allclose(out_off, hidden) == True | ✅ PASS |
| **M6 开启 ↔ 不同 metadata 下 logits 改变** | stub.enabled=True → out != hidden, 不同 metadata 下 out 不同 | torch.allclose 双重验证 | ✅ PASS |
| **M7 metadata gradient 路径** | stub.proj.weight.grad 非零 | backward() 后 stub.proj.weight.grad.max = 1.25e+01 | ✅ PASS |

→ **7/7 markers 全部 PASS, Issue #74 预检完整 (R18 实证)**.

## 3. 实施 + 验证 (R19 强制)

### 3.1 真实数据 (R18 强制)

实施组件:
- `SIDMetadata` dataclass (M1 schema)
- `serialize_metadata(metadata_list, device='cuda:0')` (M2)
- `deserialize_metadata(serialized)` (M3)
- `AttentionBiasStub(nn.Module)` (M4, M5, M6)
  - 关闭时: outputs = hidden_states (M5 等价 vanilla T5)
  - 开启时: outputs = hidden_states + bias (bias = proj(kappa_l * scale_l)) (M6)
- gradient 验证: `out_on_same.sum().backward()` → stub.proj.weight.grad 非零 (M7)

### 3.2 修复路径

- 第一次实施有 device mismatch (CPU tensor vs CUDA stub): 修复 serialize_metadata 接受 device 参数
- 第一次实施有 shape mismatch (bias (3, 512) vs hidden (2, 10, 512)): 修复 bias = bias.mean(dim=0).unsqueeze(0) 广播
- 最终运行: 7/7 markers 全部 PASS

## 4. 产物清单

| 路径 | 内容 |
|------|------|
| descriptions/task367_issue74_direction_c_precheck_sid_metadata.md | 本 description |
| verdicts/task367_issue74_direction_c_precheck_sid_metadata_v2.md | 本 verdict (R18 实证) |
| scripts/task367_issue74_sid_metadata_implementation.py | sid_metadata + attention-bias stub 实施 |
| logs/task367_issue74_sid_metadata/training.log | 7 markers 验证 log |
| products/task367_issue74_sid_metadata/_TRAINING_PID | PID file |

---

result: Issue #74 [方向C 预检 sid_metadata] 预检 7/7 markers 全部 PASS (R18 强制实证). 实施 SIDMetadata dataclass + serialize/deserialize + AttentionBiasStub (可关闭 ↔ vanilla T5 等价). 真实数据: stub.enabled=False → out == hidden (M5 PASS), stub.enabled=True + 不同 metadata → out 不同 (M6 PASS), stub.proj.weight.grad 非零 max=1.25e+01 (M7 PASS). 跟 #71/#68/#65 同路径 Stage3 vanilla T5, 但 #74 给出 7 markers 实施路径, 实证成功. Gate 1=Stage1 ⏸ STOP (预检任务), Gate 1.5=预检 7/7 ✅ PASS, Gate 2/3/4 ⏸ STOP per spec.
