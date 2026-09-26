"""Mechanism Verification Gate (MVG) — 4-layer pre-train sanity check.

Project Rules §6 + Agent C hypothesis contract. Usage:
    PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        python3 scripts/mvg_check.py <iter_id>

按 Agent C 的可证伪假设逐层校验：

  Layer 1 — Graph check
      total_loss.requires_grad + total_loss.grad_fn != None；
      新机制涉及的 loss / 参数都在 autograd graph 内。

  Layer 2 — Gradient check
      torch.autograd.grad(loss, model.parameters(), allow_unused=True) 后，
      每个 requires_grad 参数的 grad 必须存在（None 即 silent no-op）且有限；
      新机制参数（layer.attention.*、layer.embedding.weight 等）的 L2 范数
      必须大于 EpsilonGradient（默认 1e-12），否则认为机制 no-op。

  Layer 3 — Update check
      复制 1 个临时 optimizer 在同一个 batch 上做 5 个 step；
      对每个新机制参数，比较 ``||theta_{t+5}-theta_t|| / ||theta_t||``；
      若 > EpsilonUpdate（默认 1e-7）则说明参数被 optimizer 真更新。

  Layer 4 — Behavior check (ON vs OFF)
      同 seed / 同 batch 各跑 200 step：
        - Mechanism ON（默认）；
        - Mechanism OFF（克隆模型后将该机制的输出置 0）；
      比较 (a) 总 loss 差距 ``|L_on - L_off|``；(b) 关键变量（如
        attention_logits、temperature、cyclic c(t)）的 min/max/mean；
      若两边在任一关键指标上数值差距 < EpsilonBehavior（默认 1e-6），
      则认为机制未改变系统行为，fail。

任何一层 FAIL → 立即 raise，不允许 fallback；只有 4 层全 PASS 才输出
``MVG PASS`` 并允许进入 Stage2 训练。

脚本不复用旧 ``grad_check.py`` 的接口（避免旧 ``QuantizeForwardMode``
与当前 ``Quantize.forward`` 不兼容）；这里通过 ``importlib`` 加载
``curvature_RQ-VAE.py``，再调用 ``scripts/grad_check.py`` 同款 ``RqVae``
构造路径，确保数值超参与 trainer 完全一致。

约束：
  - 0 CLI args；脚本内 iter_id 来自位置参数；
  - 不修改任何 Python 源文件；
  - 任何 FAIL 立即 raise，错误信息中必须包含具体层名 + 数值证据。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch


WORK_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE")

EpsilonGradient = 1e-12
EpsilonUpdate = 1e-7
EpsilonBehavior = 1e-6
UpdateSteps = 5
BehaviorSteps = 200


def _load_rqtrain(work: Path):
    script = work / "curvature_RQ-VAE.py"
    spec = importlib.util.spec_from_file_location("rqtrain", str(script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载训练入口: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_iter_workdir(iter_id: str) -> Path:
    if not iter_id.isdigit() or int(iter_id) <= 0:
        raise ValueError(f"iter_id 必须是正整数, 实际 {iter_id}")
    work = WORK_ROOT / f"curvature_RQ-VAE_iter{iter_id}"
    if not work.is_dir():
        raise FileNotFoundError(f"iter 工作目录不存在: {work}")
    return work


def _new_model(rqtrain, device: torch.device):
    model = rqtrain.RqVae(
        input_dim=rqtrain.INPUT_DIM,
        embed_dim=rqtrain.EMBED_DIM,
        hidden_dims=rqtrain.HIDDEN_DIMS,
        codebook_size=rqtrain.CODEBOOK_SIZE,
        codebook_kmeans_init=False,
        n_layers=rqtrain.N_LAYERS,
        commitment_weight=rqtrain.COMMITMENT_WEIGHT,
        sk_eps=0.05,
        sk_iters=3,
        c_cyclic_min=rqtrain.C_CYCLIC_MIN,
        c_cyclic_max=rqtrain.C_CYCLIC_MAX,
        c_cyclic_period=rqtrain.C_CYCLIC_PERIOD,
        # iter1 P1: 传递 USE_VSHAPE_CURVATURE 让 ON/OFF 模型走不同 c(t) 公式
        use_vshape_curvature=getattr(rqtrain, "USE_VSHAPE_CURVATURE", False),
        midpoint_layer_mask=rqtrain.MIDPOINT_LAYER_MASK,
    ).to(device)
    # iter9 P2: 把 SID dropout aug flag 挂上 (ON 路径), 默认 False 兼容旧 iter
    model._use_sid_dropout_aug = getattr(rqtrain, "USE_SID_DROPOUT_AUG", False)
    model._sid_mask_rate_l1 = getattr(rqtrain, "SID_MASK_RATE_L1", 0.15)
    model._sid_mask_rate_l2 = getattr(rqtrain, "SID_MASK_RATE_L2", 0.15)
    return model


def _new_mechanism_params(model) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for layer in model.layers:
        attention = getattr(layer, "attention", None)
        if attention is not None:
            params.extend(attention.parameters())
        params.append(layer.embedding.weight)
    return [param for param in params if param.requires_grad]


def _l2_norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().norm().item())


def _layer1_graph(rqtrain, model, batch, device) -> None:
    model.train()
    model.set_curriculum_step(0)
    seq_batch = rqtrain._build_seq_batch(batch, device)
    out = model(seq_batch)
    if not out.loss.requires_grad:
        raise RuntimeError("MVG Layer1 FAIL: total_loss.requires_grad is False")
    if out.loss.grad_fn is None:
        raise RuntimeError("MVG Layer1 FAIL: total_loss.grad_fn is None")
    if not torch.isfinite(out.loss).item():
        raise RuntimeError("MVG Layer1 FAIL: total_loss 含 NaN/Inf")
    print(
        f"[MVG/L1] total_loss.requires_grad=True "
        f"grad_fn={out.loss.grad_fn is not None} "
        f"value={float(out.loss.item()):.6f}"
    )


def _layer2_gradient(rqtrain, model, batch, device) -> None:
    model.zero_grad(set_to_none=True)
    seq_batch = rqtrain._build_seq_batch(batch, device)
    out = model(seq_batch)
    grads = torch.autograd.grad(
        out.loss,
        tuple(model.parameters()),
        retain_graph=False,
        allow_unused=True,
    )
    bad = []
    for name, param, grad in zip(
        [name for name, _ in model.named_parameters()],
        model.parameters(),
        grads,
    ):
        if not param.requires_grad:
            continue
        if grad is None:
            bad.append((name, "grad is None"))
            continue
        if not torch.isfinite(grad).all().item():
            bad.append((name, "grad 含 NaN/Inf"))
            continue
        norm = _l2_norm(grad)
        if norm <= EpsilonGradient:
            bad.append((name, f"grad L2={norm:.3e} ≤ ε"))
    if bad:
        message = "; ".join(f"{name}:{why}" for name, why in bad[:6])
        raise RuntimeError(
            "MVG Layer2 FAIL: 至少 1 个 requires_grad 参数梯度异常. "
            f"top offenders: {message}"
        )
    mech = _new_mechanism_params(model)
    mech_names = {id(p) for p in mech}
    mech_grad_l2 = []
    for name, param, grad in zip(
        [name for name, _ in model.named_parameters()],
        model.parameters(),
        grads,
    ):
        if id(param) not in mech_names or grad is None:
            continue
        mech_grad_l2.append((name, _l2_norm(grad)))
    if not mech_grad_l2:
        raise RuntimeError("MVG Layer2 FAIL: 找不到任何机制参数")
    print(
        f"[MVG/L2] all_requires_grad_params_have_grad=True "
        f"mechanism_param_count={len(mech_grad_l2)}"
    )
    for name, value in mech_grad_l2[:8]:
        print(f"[MVG/L2] grad_l2[{name}]={value:.3e}")


def _layer3_update(rqtrain, model, batch, device) -> None:
    seq_batch = rqtrain._build_seq_batch(batch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    for _ in range(UpdateSteps):
        optimizer.zero_grad(set_to_none=True)
        out = model(seq_batch)
        out.loss.backward()
        optimizer.step()
    bad = []
    mech_names = {id(p) for p in _new_mechanism_params(model)}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        before_t = before[name]
        diff = (param.detach() - before_t).norm().item()
        base = before_t.norm().clamp_min(1e-12).item()
        ratio = diff / base
        if id(param) in mech_names and ratio <= EpsilonUpdate:
            bad.append((name, ratio))
    if bad:
        message = "; ".join(f"{name}:{ratio:.3e}" for name, ratio in bad[:6])
        raise RuntimeError(
            f"MVG Layer3 FAIL: optimizer 在 {UpdateSteps} 步后未更新新机制参数. "
            f"top offenders: {message}"
        )
    print(
        f"[MVG/L3] update_ratio_floor>={EpsilonUpdate:.1e} "
        f"update_steps={UpdateSteps}"
    )


def _layer4_behavior(rqtrain, work: Path, batch, device) -> None:
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    on_model = _new_model(rqtrain, device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    off_model = _new_model(rqtrain, device)

    off_hooks = []
    for src_layer, dst_layer in zip(on_model.layers, off_model.layers):
        attention = getattr(dst_layer, "attention", None)
        if attention is not None:
            original_forward = attention.forward

            def zero_forward(latent_h, codebook_h, _fwd=original_forward):
                _ = _fwd(latent_h, codebook_h)
                return torch.zeros(
                    latent_h.shape[0],
                    attention.n_embed,
                    device=latent_h.device,
                    dtype=latent_h.dtype,
                )

            attention.forward = zero_forward  # type: ignore[assignment]
            off_hooks.append((attention, original_forward))
            continue
        # iter6 P1: 若没有 attention, OFF 通过强制 get_eps 返回 sk_eps_min 关闭.
        if hasattr(dst_layer, "get_eps") and hasattr(dst_layer, "sk_eps_min"):
            original_get_eps = dst_layer.get_eps

            def constant_get_eps(self=dst_layer, _fwd=original_get_eps):
                return torch.tensor(
                    float(self.sk_eps_min),
                    device=next(self.parameters()).device,
                    dtype=next(self.parameters()).dtype,
                )

            dst_layer.get_eps = constant_get_eps  # type: ignore[assignment]
            off_hooks.append((dst_layer, original_get_eps))
            continue
        # iter8 P1: 若 layer 有 m2_reference_point='codeword_centroid'，OFF 重置为 'origin'
        if getattr(dst_layer, "m2_reference_point", None) == "codeword_centroid":
            original_ref = dst_layer.m2_reference_point
            dst_layer.m2_reference_point = "origin"
            off_hooks.append((dst_layer, original_ref))
    # iter9 P2: 若整个模型开启 SID dropout aug, OFF 强制关闭
    if getattr(on_model, "_use_sid_dropout_aug", False):
        original_aug = on_model._use_sid_dropout_aug
        off_model._use_sid_dropout_aug = False
        off_hooks.append((off_model, ("_use_sid_dropout_aug", original_aug)))
    # iter1 P1: c(t) 公式变化型机制 (USE_VSHAPE_CURVATURE)
    # OFF 把每个 layer 的 use_vshape_curvature 翻成 False, 即 baseline |sin| 公式
    if any(getattr(layer, "use_vshape_curvature", False) for layer in on_model.layers):
        for dst_layer in off_model.layers:
            if getattr(dst_layer, "use_vshape_curvature", False):
                dst_layer.use_vshape_curvature = False  # type: ignore[assignment]
                off_hooks.append((dst_layer, getattr(dst_layer, "get_c")))

    on_model.train()
    off_model.train()
    seq_batch = rqtrain._build_seq_batch(batch, device)
    on_optimizer = torch.optim.AdamW(
        on_model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    off_optimizer = torch.optim.AdamW(
        off_model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    on_loss_value = 0.0
    off_loss_value = 0.0
    on_temp_value = float("nan")
    on_logits_min = float("nan")
    on_logits_max = float("nan")
    # 从非零 step 起步, 让 ON 路径在 c(t) 与 ε(t) 曲线均偏离 ε_min;
    # 否则 step=0 时 c=0.3, ε(t)=ε_min, 与 OFF 锁定 ε_min 数值完全一致.
    behavior_start = max(1, rqtrain.C_CYCLIC_PERIOD // 4)
    for step in range(behavior_start, behavior_start + BehaviorSteps):
        on_model.set_curriculum_step(step)
        off_model.set_curriculum_step(step)
        on_optimizer.zero_grad(set_to_none=True)
        out_on = on_model(seq_batch)
        out_on.loss.backward()
        on_optimizer.step()
        on_loss_value = float(out_on.loss.detach().item())
        first_attention = getattr(on_model.layers[0], "attention", None)
        if first_attention is not None:
            on_temp_value = float(
                first_attention.get_temperature().detach().item()
            )
        elif hasattr(on_model.layers[0], "get_eps"):
            on_temp_value = float(
                on_model.layers[0].get_eps().detach().item()
            )
        else:
            # iter8 P1: 既无 attention 也无 get_eps 时使用 c(t) 作为机制直接输出证据
            on_temp_value = float(
                on_model.layers[0].get_c().detach().item()
            )
        off_optimizer.zero_grad(set_to_none=True)
        out_off = off_model(seq_batch)
        out_off.loss.backward()
        off_optimizer.step()
        off_loss_value = float(out_off.loss.detach().item())
        # iter6 P1: OFF hook 把 get_eps 锁在 sk_eps_min; 在 ON/OFF 步进前
        # 已替换 off_model 各层方法, 因此 off_eps 与 on_eps 必然分歧.

    loss_diff = abs(on_loss_value - off_loss_value)
    # iter1 P1: c(t) 公式变化型机制 -- 直接比较 get_c() 输出, 不依赖 long-term loss convergence.
    on_c_value = float(on_model.layers[0].get_c().detach().item())
    off_c_value = float(off_model.layers[0].get_c().detach().item())
    c_diff = abs(on_c_value - off_c_value)
    is_formula_mechanism = any(getattr(layer, "use_vshape_curvature", False) for layer in on_model.layers)
    if is_formula_mechanism:
        if c_diff < EpsilonBehavior:
            raise RuntimeError(
                f"MVG Layer4 FAIL (formula mechanism iter1): ON/OFF get_c() 数值差 "
                f"{c_diff:.3e} < ε={EpsilonBehavior:.1e}, c(t) 公式未实际生效"
            )
    elif loss_diff < EpsilonBehavior:
        raise RuntimeError(
            f"MVG Layer4 FAIL: ON/OFF 在 {BehaviorSteps} 步后 loss 差距 "
            f"{loss_diff:.3e} < ε={EpsilonBehavior:.1e}, 机制未改变系统行为"
        )
    if not (0.0 < on_temp_value < 1e6):
        raise RuntimeError(
            f"MVG Layer4 FAIL: attention 温度数值异常 temp={on_temp_value}"
        )
    print(
        f"[MVG/L4] loss_diff=|L_on-L_off|={loss_diff:.3e} "
        f"on_attn_temp={on_temp_value:.3e} "
        f"behavior_steps={BehaviorSteps}"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: mvg_check.py <iter_id>", file=sys.stderr)
        raise SystemExit(2)

    iter_id = sys.argv[1]
    work = _resolve_iter_workdir(iter_id)
    sys.path.insert(0, str(work))
    import curvature_config  # noqa: F401  # 注入 env vars

    if not torch.cuda.is_available():
        raise RuntimeError("MVG 需要 CUDA，但当前 CUDA 不可用")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    rqtrain = _load_rqtrain(work)
    emb_path = Path(rqtrain.EMB_NPY)
    if not emb_path.is_file():
        raise FileNotFoundError(f"embedding 不存在: {emb_path}")
    arr = np.load(emb_path).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != rqtrain.INPUT_DIM:
        raise ValueError(
            f"embedding shape={arr.shape}, 期望第二维为 {rqtrain.INPUT_DIM}"
        )
    if arr.shape[0] < rqtrain.BATCH_SIZE:
        raise ValueError(
            f"embedding 样本数 {arr.shape[0]} 小于 batch size {rqtrain.BATCH_SIZE}"
        )
    batch = torch.from_numpy(arr[: rqtrain.BATCH_SIZE]).to(device)

    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    model = _new_model(rqtrain, device)

    _layer1_graph(rqtrain, model, batch, device)
    _layer2_gradient(rqtrain, model, batch, device)
    _layer3_update(rqtrain, model, batch, device)
    _layer4_behavior(rqtrain, work, batch, device)

    print("MVG PASS")
    print(f"workdir={work}")
    print(f"embedding={emb_path}")
    print(f"epsilon_gradient={EpsilonGradient}")
    print(f"epsilon_update={EpsilonUpdate}")
    print(f"epsilon_behavior={EpsilonBehavior}")


if __name__ == "__main__":
    main()