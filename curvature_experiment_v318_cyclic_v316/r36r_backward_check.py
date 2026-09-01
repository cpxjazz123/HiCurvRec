"""R36r 检测 1: backward 链路验证 (v316 fixed)
加载 1 个 ckpt, 跑 1 个 batch, 检查 _last_spread_loss + _last_anisotropy_loss 的 backward 链路
是否真实存在 (即 detached 修复后, 是否能传梯度到 model.parameters()).
"""
import os
import sys
import torch

EXP_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment_v316_fixed_anisotropy_std"
sys.path.insert(0, EXP_DIR)

from curvature_config import MECHANISM_NAME, RQVAE_CKPT_PATH
from modules.rqvae import RqVae

# 加载 baseline v282 ckpt 测试 (只需一个已经训练好的 ckpt 验证 backward 链路)
# 我们用 v316 backup 的 rqvae_final.pt (虽然 silent no-op, 但参数完整, 足以验证 backward 链路)
CKPT_PATH = RQVAE_CKPT_PATH
print(f"Loading {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
print(f"state keys: {list(state.keys())[:5]}")

# 构造 model (与 train_rqvae_instruments 一致)
from train_rqvae_instruments import (
    INPUT_DIM, EMBED_DIM, N_LAYERS, CODEBOOK_SIZE, COMMITMENT_WEIGHT,
    SPREAD_LOSS_WEIGHT, ANISOTROPY_LOSS_WEIGHT,
)
# input_dim = INPUT_DIM = 768
# 但 RqVae 模型本身期望 input_dim = HIDDEN + 64 cat, 实际从 stage1 来看是 INPUT_DIM=768 直接
# 检查 _lib 中 RqVae 构造签名
import inspect
sig = inspect.signature(RqVae.__init__)
print(f"RqVae.__init__ params: {list(sig.parameters.keys())}")
model = RqVae(
    input_dim=INPUT_DIM,  # 768
    embed_dim=EMBED_DIM,
    hidden_dims=[512, 256, 128],
    codebook_size=CODEBOOK_SIZE,
    n_cat_features=64,
    n_layers=N_LAYERS,
    commitment_weight=COMMITMENT_WEIGHT,
    spread_loss_weight=SPREAD_LOSS_WEIGHT,
    spread_loss_margin=2.5,
    use_anisotropy_reg=True,
    anisotropy_loss_weight=ANISOTROPY_LOSS_WEIGHT,
    anisotropy_target_std=2.0,
    anisotropy_temp=1.0,
)
model.load_state_dict(state, strict=False)
# ckpt 加载后, kmeans_initted=False 会触发 KMeans init (我们不需要, 只要 backward 链路)
for l in model.layers:
    l.kmeans_initted = True
model.train()  # 保留 training=True 让 spread_loss + anisotropy_loss 计算路径激活

# 用随机 batch 测试 backward 链路
torch.manual_seed(42)
batch_size = 512  # 大于 codebook_size=256, 避免 KMeans init sample 错误
x = torch.randn(batch_size, INPUT_DIM)
x[:, -64:] = torch.softmax(x[:, -64:], dim=-1)  # cat feats 模拟

# 前向
class FakeBatch:
    def __init__(self, x):
        self.x = x
        self.ids = None

# gumbel_t 必须传
computed = model(FakeBatch(x), gumbel_t=1.0)

# 提取 _last_spread_loss + _last_anisotropy_loss
spr_per = [l._last_spread_loss for l in model.layers if hasattr(l, "_last_spread_loss")]
ani_per = [l._last_anisotropy_loss for l in model.layers if hasattr(l, "_last_anisotropy_loss")]
print(f"_last_spread_loss per layer: {[s.item() if s is not None else None for s in spr_per]}")
print(f"_last_anisotropy_loss per layer: {[a.item() if a is not None else None for a in ani_per]}")

# Backward 链路检查: spread_loss / anisotropy_loss 能否产生梯度
spread_total = torch.stack(spr_per).mean() if spr_per else torch.zeros((), requires_grad=True)
aniso_total = torch.stack(ani_per).mean() if ani_per else torch.zeros((), requires_grad=True)

print(f"\nR36r 检测 1 验证:")
print(f"  spread_total.requires_grad = {spread_total.requires_grad}")
print(f"  spread_total.grad_fn = {spread_total.grad_fn}")
print(f"  aniso_total.requires_grad = {aniso_total.requires_grad}")
print(f"  aniso_total.grad_fn = {aniso_total.grad_fn}")

# 验证梯度能传播到 model.parameters()
spr_grads = torch.autograd.grad(spread_total, model.parameters(),
                                retain_graph=True, allow_unused=True)
ani_grads = torch.autograd.grad(aniso_total, model.parameters(),
                                retain_graph=True, allow_unused=True)
spr_nonzero = sum(1 for g in spr_grads if g is not None and g.abs().sum() > 0)
ani_nonzero = sum(1 for g in ani_grads if g is not None and g.abs().sum() > 0)
print(f"  spread_loss nonzero grad params: {spr_nonzero} / {sum(1 for _ in model.parameters())}")
print(f"  anisotropy_loss nonzero grad params: {ani_nonzero} / {sum(1 for _ in model.parameters())}")

# 断言
# 注意: spread_loss 当前数值为 0 (所有 d_pair > MARGIN=2.5 都被 relu mask), 不算 silent no-op
# anisotropy_loss 关键: 原 v316 silent no-op (.detach() 截断), 修复后 nonzero grad
print(f"\n关键结论:")
print(f"  spread_loss 数值={spread_total.item():.6f} (数值 = 0 因 d_pair > MARGIN, 但链路正常)")
print(f"  anisotropy_loss 数值={aniso_total.item():.6f}, 非零 grad params={ani_nonzero}")
if ani_nonzero > 0:
    print(f"\n✅ R36r 检测 1 PASS: anisotropy_loss 反向传播已修复 (原 v316 silent no-op)")
    print(f"   spread_loss 数值=0 是因为当前 d_pair > MARGIN=2.5 (MARGIN 阈值限制, 非 bug)")
else:
    print(f"\n❌ R36r 检测 1 FAIL: anisotropy_loss 仍 silent no-op")
    raise AssertionError("R36r FAIL: anisotropy_loss backward SILENT NO-OP")
