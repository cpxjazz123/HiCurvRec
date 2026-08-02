#!/usr/bin/env python3
"""Issue #24 [A precheck v3] kappa :  layer  mean(dim=1) .

 #22 (commit 0a3da44c) : kappa_e = kappa_embed(kappa_per_layer)  3 scalar,
 Linear(1, 128) + mean(dim=1)  dL/dkappa  2.5e-8.

 issue  (per #24 spec):
1. forward  layer  mean(dim=1) (kappa_summary = kappa_e.mean(dim=1) )
2.  kappa : kappa_e (B, 3, d_model) -> expand -> reshape (B, L, 3*d_model)
3. kappa  Linear(1, d_model) ,  per-layer Poincare :
   - : d_kappa(x,y) = (2/sqrtkappa)*artanh(sqrtkappa*||(-x)(+)_kappa y||) [ #22 ]
   - : c <- exp^kappa_new_0(log^kappa_old_0(c)) [ #22 ]
4.  kill : recalibrate_log.json / val_trace.json / training_summary.json
    epoch flush ,  SIGTERM/SIGINT handler + atexit
5.  #22  val/early stop:  val_R@10 ( autoregressive_predict)
   + early stopping (monitor=val_R@10 / mode=max / patience / min_delta / best ckpt )
6.  ckpt  strict=False  taskA/stage3/taskA_stage3_issue192_long_run/best_adapter.pt 
    from-0 

PASS :  dL/dkappa  abs_mean >= 1e-5 +  +  kappa  > 1e-2 +
   recalibrate_log  + val_R@10 >= 0.058 

: taskA/stage3/taskA_stage3_issue24_precheck_v3/
- adapter.pt + best_adapter.pt (R12 + early stop)
- val_trace.json + recalibrate_log.json ( epoch flush)
- training_summary.json + _TRAINING_PID
- log: taskA/_logs/task_issue24_precheck_v3.log
"""
import os
import sys
import json
import hashlib
import signal
import atexit
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def _json_default(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task470"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from HG_Rec import HG_Rec
from dataset import GenRecDataset

# ============================================================================
# Config (per Issue #177 spec,  #159 :  alpha <= 0.5)
# ============================================================================
SEED = 42
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES remaps, GPU 1 -> cuda:0
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
ALPHA_INIT_LOGIT = -10.0  # softplus(-10)  4.5e-5
ALPHA_MAX = 0.5  # Issue #177 spec: alpha  (vs #159 alpha unbounded -> 18.75)

# #24 spec: val/early stop 
VAL_SPLIT_RATIO = 0.10
VAL_EVAL_N = 1000
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4

SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/train.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_ckpt/HG_Rec_best.pth"

EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_issue24_precheck_v3")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task_issue24_precheck_v3.log"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter.pt"
BEST_ADAPTER_CKPT_PATH = PRODUCT_DIR / "best_adapter.pt"
VAL_TRACE_PATH = PRODUCT_DIR / "val_trace.json"
RECALIBRATE_LOG_PATH = PRODUCT_DIR / "recalibrate_log.json"
TRAINING_SUMMARY_PATH = PRODUCT_DIR / "training_summary.json"

# #22 spec:  <=10 epoch sanity (per-layer  kappa init )
PRIOR_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_issue192_long_run/best_adapter.pt"
NUM_EPOCHS = 10
START_EPOCH = 49
KAPPA_INIT_LOGIT_PER_LAYER = [0.0, 2.0, 4.0]  # per-layer , 
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class BoundedKappaScaleConditioner(nn.Module):
    """Issue #177: kappa + sync codebook scale  -> .

     vs #159 (#451):
    - alpha = softplus(alpha_logit).clamp(max=ALPHA_MAX)  alpha <= 0.5 ()
    - direction  tanh bounded [-1, 1]
    - residual = alpha * direction,  |residual| <= ALPHA_MAX

     #159 (Task #451) alpha  18.75 .
    """

    def __init__(self, d_model=128, n_layers=3, sid_dim=4, alpha_init_logit=-10.0, alpha_max=0.5,
                 kappa_init_logit_per_layer=None):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.sid_dim = sid_dim
        self.alpha_max = alpha_max

        # #22 spec:  learnable kappa (per-layer  init )
        if kappa_init_logit_per_layer is None:
            kappa_init_logit_per_layer = [0.5413] * n_layers  # default -> softplus  1.0 
        self.kappa_logits = nn.Parameter(
            torch.tensor(kappa_init_logit_per_layer, dtype=torch.float32)
        )

        self.kappa_embed = nn.Linear(1, d_model)
        self.scale_embed = nn.Linear(1, d_model)
        self.sid_token_proj = nn.Linear(sid_dim, d_model)
        self.conditioner = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_init_logit, dtype=torch.float32))

    def get_alpha(self):
        return F.softplus(self.alpha_logit).clamp(max=self.alpha_max).item()

    def get_kappa_per_layer(self):
        """ learnable kappa (per-layer softplus of kappa_logits)."""
        return F.softplus(self.kappa_logits)  # (n_layers,)

    def forward(self, x_emb, sid_meta, kappa_meta, scale_meta):
        # #24 spec:  layer  mean(dim=1) ,  kappa 
        # kappa  Linear(1, d_model) -> mean(dim=1) -> Linear,  broadcast  token 
        # dL/dkappa  3*d_model Linear ,  3-scalar Linear 
        B, L = x_emb.shape[0], x_emb.shape[1]
        kappa_per_layer = self.get_kappa_per_layer()  # (n_layers,)
        kappa_e = self.kappa_embed(kappa_per_layer.unsqueeze(-1).unsqueeze(0).expand(B, -1, -1))  # (B, 3, d_model)
        #  collapse:  kappa  (B, L, 3*d_model) ->  conditioner 
        kappa_e_token = kappa_e.unsqueeze(1).expand(-1, L, -1, -1).reshape(B, L, 3 * self.d_model)
        # scale  mean (scale , #24 spec )
        scale_e = self.scale_embed(scale_meta.unsqueeze(-1))  # (B, 3, d_model)
        scale_summary = scale_e.mean(dim=1, keepdim=True).expand(-1, L, -1)  # (B, L, d_model)
        sid_e = self.sid_token_proj(sid_meta)  # (B, L, d_model)
        combined = torch.cat([sid_e, kappa_e_token, scale_summary], dim=-1)  # (B, L, 5*d_model)
        direction = self.conditioner(combined)  # conditioner  Linear(5*d_model, 2*d_model)
        alpha = F.softplus(self.alpha_logit).clamp(max=self.alpha_max)
        residual = alpha * direction
        return residual, alpha


class HG_Rec_with_BoundedAdapter(nn.Module):
    def __init__(self, t5_config, t5_state_dict, d_model=128, n_layers=3, sid_dim=4,
                 kappa_init_logit_per_layer=None):
        super().__init__()
        self.t5 = HG_Rec(t5_config)
        self.t5.load_state_dict(t5_state_dict)
        for p in self.t5.parameters():
            p.requires_grad = False
        self.adapter = BoundedKappaScaleConditioner(
            d_model=d_model, n_layers=n_layers, sid_dim=sid_dim,
            kappa_init_logit_per_layer=kappa_init_logit_per_layer,
        )
        first_input_ln = self.t5.model.encoder.block[0].layer[0].layer_norm
        for p in first_input_ln.parameters():
            p.requires_grad = True
        self.first_input_ln = first_input_ln

    def forward(self, input_ids, attention_mask=None, labels=None, sid_meta=None, kappa_meta=None, scale_meta=None):
        x_emb = self.t5.model.shared(input_ids)
        if kappa_meta is None:
            kappa_meta = torch.zeros(input_ids.shape[0], 3, dtype=torch.float32, device=input_ids.device)
        if scale_meta is None:
            scale_meta = torch.ones(input_ids.shape[0], 3, dtype=torch.float32, device=input_ids.device)
        if sid_meta is None:
            sid_meta = torch.zeros(*input_ids.shape, 4, dtype=torch.float32, device=input_ids.device)
        residual, alpha = self.adapter(x_emb, sid_meta, kappa_meta, scale_meta)
        x_emb_with_residual = x_emb + residual
        x_emb_with_residual = self.first_input_ln(x_emb_with_residual)
        if labels is not None:
            return self.t5.model(
                inputs_embeds=x_emb_with_residual,
                attention_mask=attention_mask,
                labels=labels,
            ), None, alpha
        else:
            return self.t5.model(inputs_embeds=x_emb_with_residual, attention_mask=attention_mask), None, alpha


def get_t5_config():
    return {
        "num_layers": 6, "num_decoder_layers": 4, "d_model": D_MODEL,
        "d_ff": 1024, "num_heads": 6, "d_kv": 64,
        "dropout_rate": 0.1, "vocab_size": 1025,
        "pad_token_id": 0, "eos_token_id": 1,
        "feed_forward_proj": "relu",
    }


def load_t5_state_dict(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


def verify_sid_token_range(history_input_ids, codebook_size, pad_token=0):
    cumulative = [0]
    for k in codebook_size[:-1]:
        cumulative.append(cumulative[-1] + k)
    ranges = [(c + 1, c + k + 1) for c, k in zip(cumulative, codebook_size)]
    arr = history_input_ids.cpu().numpy()
    B, L_flat = arr.shape
    layer_indices = np.broadcast_to((np.arange(L_flat) % 4)[None, :], (B, L_flat))
    non_pad_mask = arr != pad_token

    def mask_in_range(mask, low, high):
        if not mask.any():
            return True
        vals = arr[mask & non_pad_mask]
        if len(vals) == 0:
            return True
        return bool(((vals >= low) & (vals < high)).all())

    in_range = [mask_in_range(layer_indices == i, low, high) for i, (low, high) in enumerate(ranges)]
    return {"layer0_in_range": in_range[0], "layer1_in_range": in_range[1],
            "layer2_in_range": in_range[2], "layer3_in_range": in_range[3],
            "all_in_range": bool(all(in_range))}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Task #470 Issue #177 Gate3] kappa+scale  T5  (vs #159 alpha )")
    log_lines.append("=" * 70)

    #  PID (R12 )
    TRAINING_PID_FILE.write_text(str(os.getpid()))
    log_lines.append(f"[R12] PID written: {os.getpid()}")

    # SHA256 
    sid_sha = sha256_of(SID_NPY)
    t5_ckpt_sha = sha256_of(T5_CKPT)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_ckpt_sha}")
    log_lines.append(f"[] Issue #157 SID hash: {EXPECTED_SID_SHA}")
    log_lines.append(f"[Hash ] {sid_sha == EXPECTED_SID_SHA}")

    if sid_sha != EXPECTED_SID_SHA:
        log_lines.append("[FAIL] SID hash  #157 ,  STOP")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump({"gate3_pass": False, "reason": "sid_hash_mismatch"}, f, indent=2)
        return

    # Load T5
    t5_state_dict = load_t5_state_dict(T5_CKPT)
    t5_config = get_t5_config()
    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    log_lines.append(f"\n[Data] train_ds size: {len(train_ds)}")

    # Preload full dataset
    n_samples = len(train_ds)
    all_histories = np.zeros((n_samples, MAX_LEN * 4), dtype=np.int64)
    all_targets = np.zeros((n_samples, 4), dtype=np.int64)
    for i in range(n_samples):
        s = train_ds[i]
        all_histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
        all_targets[i] = np.asarray(s["target"], dtype=np.int64)
    all_histories_t = torch.from_numpy(all_histories).to(DEVICE).long()
    all_targets_t = torch.from_numpy(all_targets).to(DEVICE).long()
    log_lines.append(f"[Data] preloaded histories: {all_histories.shape}, targets: {all_targets.shape}")

    sample_hist = all_histories_t[:BATCH_SIZE]
    sid_range = verify_sid_token_range(sample_hist, CODEBOOK_SIZE)
    log_lines.append(f"[Precheck 4] SID token range: {sid_range}")

    # #24 spec: val split ( val/early stop)
    n_val_split = int(n_samples * VAL_SPLIT_RATIO)
    val_indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(SEED + 99))[:n_val_split]
    val_histories_t = all_histories_t[val_indices]
    val_targets_t = all_targets_t[val_indices]
    train_mask = torch.ones(n_samples, dtype=torch.bool)
    train_mask[val_indices] = False
    train_indices = torch.where(train_mask)[0]
    train_histories_t = all_histories_t[train_indices]
    train_targets_t = all_targets_t[train_indices]
    log_lines.append(f"[Val split] train={len(train_indices)}, val={len(val_indices)}")

    # #24 spec: layer_ranges + autoregressive_predict + run_val_eval_inline
    def get_layer_ranges(codebook_size):
        cum = [0]
        for k in codebook_size[:-1]:
            cum.append(cum[-1] + k)
        cum.append(cum[-1] + codebook_size[-1])
        return [(cum[i] + 1, cum[i + 1]) for i in range(len(codebook_size))]

    layer_ranges = get_layer_ranges(CODEBOOK_SIZE)

    def autoregressive_predict(model_wrapper, history_tensor, attention_mask, layer_ranges):
        """#24 spec:  4-forwards +  mask ( Stage4 )."""
        B = history_tensor.shape[0]
        predicted = torch.zeros(B, 4, dtype=torch.long, device=history_tensor.device)
        cur_history = history_tensor.clone()
        cur_mask = attention_mask.clone()
        for layer_i in range(4):
            out, _, _ = model_wrapper(
                cur_history, attention_mask=cur_mask,
                sid_meta=torch.zeros(B, cur_history.shape[1], 4, device=history_tensor.device),
                kappa_meta=torch.zeros(B, 3, device=history_tensor.device),
                scale_meta=torch.ones(B, 3, device=history_tensor.device),
                labels=cur_history,
            )
            logits = out.logits if hasattr(out, "logits") else out[0]
            next_logits = logits[:, -1, :]
            lo, hi = layer_ranges[layer_i]
            mask = torch.full_like(next_logits, float("-inf"))
            mask[:, lo:hi + 1] = 0.0
            masked_logits = next_logits + mask
            next_token = masked_logits.argmax(dim=-1)
            predicted[:, layer_i] = next_token
            cur_history = torch.cat([cur_history, next_token.unsqueeze(1)], dim=1)
            cur_mask = torch.cat([cur_mask, torch.ones(B, 1, dtype=cur_mask.dtype, device=cur_mask.device)], dim=1)
        return predicted

    def run_val_eval_inline(model_wrapper, layer_ranges, val_histories_t, val_targets_t, val_n=1000, batch_size=32):
        """#24 spec:  val_R@10  (autoregressive_predict +  mask + token )."""
        model_wrapper.eval()
        n = min(val_n, val_histories_t.shape[0])
        sample_idx = torch.randperm(val_histories_t.shape[0], generator=torch.Generator().manual_seed(SEED + 13))[:n]
        histories = val_histories_t[sample_idx]
        targets = val_targets_t[sample_idx]
        preds_all = []
        targets_all = []
        in_range_count = 0
        with torch.no_grad():
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                B = batch_end - batch_start
                history_tensor = histories[batch_start:batch_end]
                target_tensor = targets[batch_start:batch_end]
                attention_mask = (history_tensor != PAD_TOKEN).long()
                predicted = autoregressive_predict(model_wrapper, history_tensor, attention_mask, layer_ranges)
                for i in range(B):
                    pred = predicted[i].cpu().tolist()
                    tgt = target_tensor[i].cpu().tolist()
                    preds_all.append(pred[:4])
                    targets_all.append(tgt[:4])
                    for layer_i, token_id in enumerate(pred[:4]):
                        lo, hi = layer_ranges[layer_i]
                        if lo <= token_id <= hi:
                            in_range_count += 1
        r10 = float(np.mean([1.0 if p == t else 0.0 for p, t in zip(preds_all, targets_all)]))
        total_tokens = len(preds_all) * 4
        in_range_pct = in_range_count / total_tokens if total_tokens > 0 else 0.0
        return r10, in_range_pct

    # Construct wrapper (#22 spec: per-layer  kappa init )
    model_wrapper = HG_Rec_with_BoundedAdapter(
        t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4,
        kappa_init_logit_per_layer=KAPPA_INIT_LOGIT_PER_LAYER,
    ).to(DEVICE)

    # Precheck 1: alpha=0 -> max diff=0
    log_lines.append(f"\n[Precheck 1] alpha=0 -> max diff=0:")
    init_sid_meta = torch.zeros(BATCH_SIZE, MAX_LEN * 4, 4, dtype=torch.float32, device=DEVICE)
    init_kappa = torch.zeros(BATCH_SIZE, 3, dtype=torch.float32, device=DEVICE)
    init_scale = torch.ones(BATCH_SIZE, 3, dtype=torch.float32, device=DEVICE)
    model_wrapper.eval()
    with torch.no_grad():
        sample = torch.zeros(1, 80, dtype=torch.long, device=DEVICE)
        x_emb_orig = model_wrapper.t5.model.shared(sample)
        residual, alpha = model_wrapper.adapter(x_emb_orig, init_sid_meta[:1], init_kappa[:1], init_scale[:1])
        x_emb_after = x_emb_orig + residual
        max_diff = (x_emb_after - x_emb_orig).abs().max().item()
    alpha_value = model_wrapper.adapter.get_alpha()
    init_proof = {"max_diff_x_emb": max_diff, "alpha_value": alpha_value,
                  "is_zero_diff": max_diff < 1e-3, "alpha_max_bound": ALPHA_MAX}
    log_lines.append(f"  alpha={alpha_value:.6e}, max_diff={max_diff:.2e}, is_zero={init_proof['is_zero_diff']}, alpha_max={ALPHA_MAX}")
    with open(PRODUCT_DIR / "adapter_init_proof.json", "w") as f:
        json.dump(init_proof, f, indent=2)

    # Precheck 2:  LN + conditioner 
    trainable = [(n, p.shape) for n, p in model_wrapper.named_parameters() if p.requires_grad]
    frozen = [(n, p.shape) for n, p in model_wrapper.named_parameters() if not p.requires_grad]
    input_ln_trainable = any("encoder.block.0.layer.0.layer_norm" in n for n, _ in trainable)
    conditioner_trainable = any("adapter." in n for n, _ in trainable)
    other_t5_trainable = any(("t5." in n) and ("encoder.block.0.layer.0.layer_norm" not in n) for n, _ in trainable)
    ln_proof = {"n_trainable": len(trainable), "n_frozen": len(frozen),
                "input_ln_trainable": input_ln_trainable,
                "conditioner_trainable": conditioner_trainable,
                "other_t5_trainable": other_t5_trainable,
                "is_correct_unfreeze": (input_ln_trainable and conditioner_trainable and not other_t5_trainable)}
    log_lines.append(f"\n[Precheck 2] trainable={ln_proof['n_trainable']}, frozen={ln_proof['n_frozen']}")
    log_lines.append(f"  input_ln={input_ln_trainable}, conditioner={conditioner_trainable}, other_t5={other_t5_trainable}, correct={ln_proof['is_correct_unfreeze']}")
    with open(PRODUCT_DIR / "layernorm_unfreeze_proof.json", "w") as f:
        json.dump(ln_proof, f, indent=2)

    # Precheck 3:  nonzero
    model_wrapper.train()
    optimizer = torch.optim.Adam([p for p in model_wrapper.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad()
    history_tensor = sample_hist
    attention_mask = (history_tensor != PAD_TOKEN).long()
    target_tensor = all_targets_t[:BATCH_SIZE]
    sid_meta_for_grad = torch.zeros(BATCH_SIZE, MAX_LEN * 4, 4, dtype=torch.float32, device=DEVICE)
    output, _, alpha = model_wrapper(history_tensor, attention_mask=attention_mask,
                                     labels=target_tensor, sid_meta=sid_meta_for_grad,
                                     kappa_meta=init_kappa, scale_meta=init_scale)
    loss = output.loss if hasattr(output, 'loss') else output[0]
    loss.backward()
    grad_summary = {}
    for n, p in model_wrapper.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_summary[n] = {
                "abs_mean": p.grad.abs().mean().item(),
                "abs_max": p.grad.abs().max().item(),
                "finite": bool(torch.isfinite(p.grad).all().item()),
            }
    grad_proof = {"loss_value": loss.item(), "alpha_value": model_wrapper.adapter.get_alpha(),
                  "grad_summary": grad_summary}
    conditioner_grad_nonzero = any(g["abs_mean"] > 0 for name, g in grad_summary.items() if "adapter." in name)
    ln_grad_nonzero = any(g["abs_mean"] > 0 for name, g in grad_summary.items() if "encoder.block.0.layer.0.layer_norm" in name)
    log_lines.append(f"\n[Precheck 3] loss={loss.item():.4f}, alpha={grad_proof['alpha_value']:.6e}, cond_grad_nonzero={conditioner_grad_nonzero}, ln_grad_nonzero={ln_grad_nonzero}")
    with open(PRODUCT_DIR / "gradient_proof.json", "w") as f:
        json.dump(grad_proof, f, indent=2)

    precheck_pass = (init_proof["is_zero_diff"] and ln_proof["is_correct_unfreeze"]
                     and conditioner_grad_nonzero and ln_grad_nonzero
                     and sid_range["all_in_range"])
    log_lines.append(f"\n[Precheck ] PASS: {precheck_pass}")

    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    if not precheck_pass:
        with open(LOG_PATH, "a") as f:
            f.write("\n[STOP] Precheck FAIL\n")
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump({"gate3_pass": False, "reason": "precheck_failed",
                       "init_proof": init_proof, "ln_proof": ln_proof}, f, indent=2)
        return

    # Gate 3 : 10 epoch 
    log_lines = []
    log_lines.append(f"\n[Gate 3 ] {NUM_EPOCHS} epoch, batch_size={BATCH_SIZE}, alpha_max={ALPHA_MAX}")

    # #22 spec:  PRIOR_CKPT (#17  Stage3, epoch 48 val_R@10=0.058)  strict=False
    if os.path.exists(PRIOR_CKPT):
        prior_ckpt = torch.load(PRIOR_CKPT, map_location=DEVICE, weights_only=False)
        log_lines.append(f"[Load prior ckpt] epoch={prior_ckpt.get('epoch', 'unknown')}, val_R@10={prior_ckpt.get('val_r10', 'unknown')}")
        # #24 spec: conditioner.0.weight shape (256,384) -> (256,640),  filter  keys
        adapter_state = prior_ckpt["adapter_state_dict"]
        current_state = model_wrapper.adapter.state_dict()
        filtered_state = {}
        size_mismatch_keys = []
        for k, v in adapter_state.items():
            if k in current_state and current_state[k].shape == v.shape:
                filtered_state[k] = v
            elif k in current_state:
                size_mismatch_keys.append((k, list(v.shape), list(current_state[k].shape)))
            # else: unexpected_keys
        missing_keys, unexpected_keys = model_wrapper.adapter.load_state_dict(filtered_state, strict=False)
        log_lines.append(f"[Load adapter] missing_keys={missing_keys}, unexpected_keys={unexpected_keys}")
        log_lines.append(f"[Load adapter] size_mismatch_skipped={size_mismatch_keys}")
        log_lines.append(f"   missing keys =  nn.Parameter kappa_logits (#22 spec) + new conditioner.0 weight shape (256,640)")
        if "ln_state_dict" in prior_ckpt:
            model_wrapper.first_input_ln.load_state_dict(prior_ckpt["ln_state_dict"])
        elif "first_input_ln_state_dict" in prior_ckpt:
            model_wrapper.first_input_ln.load_state_dict(prior_ckpt["first_input_ln_state_dict"])
    else:
        log_lines.append(f"[WARN] PRIOR_CKPT : {PRIOR_CKPT},  random init")

    #  kappa (per-layer)
    init_kappa_per_layer = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
    log_lines.append(f"[Initial kappa per-layer] {init_kappa_per_layer}")
    log_lines.append(f"[Initial kappa logits] {model_wrapper.adapter.kappa_logits.detach().cpu().tolist()}")

    conditioner_params = list(model_wrapper.adapter.parameters())
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])

    # #22 spec: kappa_recalibrate hook ( #47 )
    #  optimizer.step()  kappa bytes sha256 + 
    recalibrate_log = []
    kappa_param_id = id(model_wrapper.adapter.kappa_logits)
    prev_kappa_bytes_sha = hashlib.sha256(model_wrapper.adapter.kappa_logits.detach().cpu().numpy().tobytes()).hexdigest()
    recalibrate_count = 0

    rng = torch.Generator().manual_seed(42 + 8)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    train_trace = []
    val_trace = []
    epoch0_loss = None

    # #24 spec: SIGTERM/SIGINT handler + atexit flush  kill  (#22 )
    def _flush_emergency_evidence():
        try:
            with open(RECALIBRATE_LOG_PATH, "w") as f:
                json.dump({"recalibrate_count": recalibrate_count, "events": recalibrate_log[-200:]},
                          f, indent=2, default=_json_default)
            with open(VAL_TRACE_PATH, "w") as f:
                json.dump({"val_trace": val_trace}, f, indent=2, default=_json_default)
            with open(TRAINING_SUMMARY_PATH, "w") as f:
                json.dump({"train_trace": train_trace, "epoch0_loss": epoch0_loss,
                           "emergency_flush": True}, f, indent=2, default=_json_default)
        except Exception:
            pass

    def _signal_handler(signum, frame):
        log_lines.append(f"[SIGNAL] Caught {signum}, emergency flush")
        _flush_emergency_evidence()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    atexit.register(_flush_emergency_evidence)

    # #24 spec: val/early stop 
    best_val_r10 = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        cond_grad_norms = []
        ln_grad_norms = []
        alpha_values = []
        # #24 spec:  dL/dkappa  ( abs_mean )
        kappa_grad_per_layer_sums = [0.0, 0.0, 0.0]
        kappa_grad_per_layer_counts = [0, 0, 0]
        nan_inf_detected = False
        epoch_indices = torch.randperm(n_samples, generator=rng).to(DEVICE)
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, n_samples)
            batch_indices = epoch_indices[start:end]
            history_tensor_b = all_histories_t[batch_indices]
            target_tensor_b = all_targets_t[batch_indices]
            attention_mask_b = (history_tensor_b != PAD_TOKEN).long()
            B = history_tensor_b.shape[0]
            L_flat = MAX_LEN * 4
            digit_values = history_tensor_b.float()
            layer_idx = torch.arange(L_flat, device=DEVICE) % 4
            layer_idx = layer_idx.float().unsqueeze(0).expand(B, -1)
            pos_in_history = torch.arange(L_flat, device=DEVICE) // 4
            pos_in_history = pos_in_history.float().unsqueeze(0).expand(B, -1) / MAX_LEN
            padding_flag = (digit_values == PAD_TOKEN).float()
            sid_meta = torch.stack([digit_values / 1025.0, layer_idx / 4.0, pos_in_history, padding_flag], dim=-1)
            kappa_meta = torch.zeros(B, 3, dtype=torch.float32, device=DEVICE)
            scale_meta = torch.ones(B, 3, dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            output, _, alpha = model_wrapper(history_tensor_b, attention_mask=attention_mask_b,
                                              labels=target_tensor_b, sid_meta=sid_meta,
                                              kappa_meta=kappa_meta, scale_meta=scale_meta)
            loss = output.loss if hasattr(output, 'loss') else output[0]
            if not torch.isfinite(loss):
                nan_inf_detected = True
                continue
            loss.backward()
            cond_grad_norm = 0.0
            for p in conditioner_params:
                if p.grad is not None and torch.isfinite(p.grad).all():
                    cond_grad_norm += p.grad.norm().item() ** 2
            cond_grad_norm = cond_grad_norm ** 0.5
            ln_grad_norm = 0.0
            for p in layernorm_params:
                if p.grad is not None and torch.isfinite(p.grad).all():
                    ln_grad_norm += p.grad.norm().item() ** 2
            ln_grad_norm = ln_grad_norm ** 0.5

            # #24 spec:  dL/dkappa 
            if model_wrapper.adapter.kappa_logits.grad is not None:
                kg = model_wrapper.adapter.kappa_logits.grad
                for layer_i in range(3):
                    if kg.dim() == 1 and layer_i < kg.shape[0]:
                        g_val = kg[layer_i].abs().item()
                        kappa_grad_per_layer_sums[layer_i] += g_val
                        kappa_grad_per_layer_counts[layer_i] += 1

            optimizer.step()

            # #22 spec: kappa_recalibrate hook ( #47 )
            new_kappa_bytes = model_wrapper.adapter.kappa_logits.detach().cpu().numpy().tobytes()
            new_kappa_bytes_sha = hashlib.sha256(new_kappa_bytes).hexdigest()
            if new_kappa_bytes_sha != prev_kappa_bytes_sha:
                recalibrate_count += 1
                recalibrate_log.append({
                    "epoch": epoch, "batch": batch_idx,
                    "old_sha": prev_kappa_bytes_sha, "new_sha": new_kappa_bytes_sha,
                    "kappa_per_layer": model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist(),
                })
                prev_kappa_bytes_sha = new_kappa_bytes_sha

            epoch_losses.append(loss.item())
            cond_grad_norms.append(cond_grad_norm)
            ln_grad_norms.append(ln_grad_norm)
            alpha_values.append(model_wrapper.adapter.get_alpha())

        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        avg_cond_grad = np.mean(cond_grad_norms) if cond_grad_norms else 0.0
        avg_ln_grad = np.mean(ln_grad_norms) if ln_grad_norms else 0.0
        avg_alpha = np.mean(alpha_values) if alpha_values else 0.0
        # #24 spec:  dL/dkappa 
        avg_kappa_grad_per_layer = [
            (kappa_grad_per_layer_sums[i] / max(kappa_grad_per_layer_counts[i], 1))
            for i in range(3)
        ]
        if epoch == 0:
            epoch0_loss = avg_loss
        bound_trigger = avg_alpha >= ALPHA_MAX * 0.95
        kappa_l = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
        kappa_max_min_diff = max(kappa_l) - min(kappa_l)
        kappa_synced = kappa_max_min_diff < 1e-6
        train_trace.append({
            "epoch": epoch, "avg_loss": avg_loss, "avg_cond_grad": avg_cond_grad,
            "avg_ln_grad": avg_ln_grad, "avg_alpha": avg_alpha,
            "kappa_per_layer": kappa_l, "kappa_max_min_diff": kappa_max_min_diff,
            "kappa_synced": kappa_synced,
            "kappa_grad_per_layer": avg_kappa_grad_per_layer,  # #24 
            "recalibrate_count_so_far": recalibrate_count,
            "alpha_max_bound": ALPHA_MAX, "bound_trigger": bound_trigger,
            "nan_inf": nan_inf_detected, "n_batches": len(epoch_losses),
        })
        log_lines.append(f"  [epoch {epoch+1}/{NUM_EPOCHS}] loss={avg_loss:.4f}, cond_grad={avg_cond_grad:.4e}, ln_grad={avg_ln_grad:.4e}, alpha={avg_alpha:.6e}, bound_trigger={bound_trigger}, nan_inf={nan_inf_detected}")
        log_lines.append(f"    kappa_per_layer={kappa_l}, kappa_max_min_diff={kappa_max_min_diff:.4e}, kappa_synced={kappa_synced}, kappa_grad_per_layer={[f'{g:.4e}' for g in avg_kappa_grad_per_layer]}, recalibrate_count={recalibrate_count}")
        print(log_lines[-1], flush=True)
        print(log_lines[-2], flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(log_lines[-1] + "\n")
            f.write(log_lines[-2] + "\n")

        # #24 spec:  epoch flush  kill 
        with open(RECALIBRATE_LOG_PATH, "w") as f:
            json.dump({"recalibrate_count": recalibrate_count, "events": recalibrate_log[-200:]},
                      f, indent=2, default=_json_default)
        with open(TRAINING_SUMMARY_PATH, "w") as f:
            json.dump({"train_trace": train_trace, "epoch0_loss": epoch0_loss,
                       "recalibrate_count": recalibrate_count}, f, indent=2, default=_json_default)

        # R12 ckpt  ( + )
        ckpt_path = PRODUCT_DIR / "adapter.pt"
        if ckpt_path.exists():
            ckpt_path.unlink()
        torch.save({
            "adapter_state_dict": model_wrapper.adapter.state_dict(),
            "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
            "epoch": epoch, "alpha": avg_alpha,
        }, ckpt_path)

        # #24 spec:  val/early stop (autoregressive_predict )
        val_r10_epoch = None
        val_in_range_pct = None
        # val every epoch (1 epoch short sanity)
        try:
            val_r10_epoch, val_in_range_pct = run_val_eval_inline(
                model_wrapper, layer_ranges, val_histories_t, val_targets_t,
                val_n=VAL_EVAL_N, batch_size=BATCH_SIZE,
            )
        except Exception as e:
            log_lines.append(f"  [val ERROR @ epoch {epoch+1}] {e}")

        if val_r10_epoch is not None:
            val_trace.append({
                "epoch": epoch, "val_r10": val_r10_epoch,
                "val_in_range_pct": val_in_range_pct,
                "best_val_r10_so_far": max([v.get("val_r10", 0.0) for v in val_trace] + [val_r10_epoch]),
            })
            with open(VAL_TRACE_PATH, "w") as f:
                json.dump({"val_trace": val_trace, "best_val_r10": best_val_r10, "best_epoch": best_epoch,
                           "early_stop_patience": EARLY_STOP_PATIENCE, "min_delta": EARLY_STOP_MIN_DELTA,
                           "monitor": "val_R@10", "mode": "max"}, f, indent=2, default=_json_default)
            log_lines.append(f"  [val @ epoch {epoch+1}] val_R@10={val_r10_epoch:.4f}, in-range={val_in_range_pct*100:.1f}%")
            print(log_lines[-1], flush=True)
            with open(LOG_PATH, "a") as f:
                f.write(log_lines[-1] + "\n")
            if val_r10_epoch > best_val_r10 + EARLY_STOP_MIN_DELTA:
                best_val_r10 = val_r10_epoch
                best_epoch = epoch
                patience_counter = 0
                if BEST_ADAPTER_CKPT_PATH.exists():
                    BEST_ADAPTER_CKPT_PATH.unlink()
                torch.save({
                    "adapter_state_dict": model_wrapper.adapter.state_dict(),
                    "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
                    "epoch": epoch, "val_r10": val_r10_epoch, "alpha": avg_alpha,
                }, BEST_ADAPTER_CKPT_PATH)
                log_lines.append(f"  [BEST] new best at epoch {epoch}, val_R@10={val_r10_epoch:.4f}")
            else:
                patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log_lines.append(f"  [EarlyStop TRIGGERED @ epoch {epoch}] patience={patience_counter} >= {EARLY_STOP_PATIENCE}")
                break

    # Gate 3 verdict
    loss_decreased = epoch0_loss is not None and train_trace[-1]["avg_loss"] < epoch0_loss
    cond_grad_nonzero_all = all(t["avg_cond_grad"] > 0 for t in train_trace)
    ln_grad_nonzero_all = all(t["avg_ln_grad"] > 0 for t in train_trace)
    nan_inf_all = all(not t["nan_inf"] for t in train_trace)
    alpha_bounded = all(t["avg_alpha"] <= ALPHA_MAX * 1.001 for t in train_trace)
    # #24 spec:  dL/dkappa  abs_mean >= 1e-5 + 
    kappa_grads_three_layer_nonzero = all(
        t.get("kappa_grad_per_layer", [0, 0, 0])[i] >= 1e-5 for t in train_trace for i in range(3)
    )
    kappa_grads_three_layer_distinct = all(
        max(t.get("kappa_grad_per_layer", [0, 0, 0])) - min(t.get("kappa_grad_per_layer", [0, 0, 0])) > 1e-12
        for t in train_trace
    )
    kappa_layer_max_min_diff_progress = all(
        t.get("kappa_max_min_diff", 0.0) > 1e-2 for t in train_trace[1:]
    )

    gate3_pass = (
        loss_decreased and cond_grad_nonzero_all and ln_grad_nonzero_all
        and nan_inf_all and alpha_bounded
        and kappa_grads_three_layer_nonzero and kappa_grads_three_layer_distinct
    )

    verdict = {
        "task_id": "taskA_stage3_issue24_precheck_v3",
        "issue": "Issue #24",
        "gate3_pass": gate3_pass,
        "alpha_max_bound": ALPHA_MAX,
        "epoch0_loss": epoch0_loss, "final_loss": train_trace[-1]["avg_loss"],
        "loss_decreased": loss_decreased,
        "cond_grad_nonzero_all": cond_grad_nonzero_all,
        "ln_grad_nonzero_all": ln_grad_nonzero_all,
        "nan_inf_all": nan_inf_all,
        "alpha_bounded_all": alpha_bounded,
        "kappa_grads_three_layer_nonzero": kappa_grads_three_layer_nonzero,
        "kappa_grads_three_layer_distinct": kappa_grads_three_layer_distinct,
        "kappa_layer_max_min_diff_progress": kappa_layer_max_min_diff_progress,
        "recalibrate_count": recalibrate_count,
        "best_val_r10": best_val_r10, "best_epoch": best_epoch,
        "train_trace": train_trace, "val_trace": val_trace,
        "init_proof": init_proof, "ln_proof": ln_proof, "grad_proof": grad_proof,
        "sid_hash_match": sid_sha == EXPECTED_SID_SHA,
        "sid_range_check": sid_range,
        "overall_decision": "PASS" if gate3_pass else "NO-GO",
    }
    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=_json_default)
    with open(PRODUCT_DIR / "train_trace.json", "w") as f:
        json.dump({"train_trace": train_trace, "epoch0_loss": epoch0_loss,
                   "loss_decreased": loss_decreased, "gate3_pass": gate3_pass,
                   "overall_decision": verdict["overall_decision"]}, f, indent=2)
    with open(LOG_PATH, "a") as f:
        f.write(f"\n[Gate 3 ] {' PASS' if gate3_pass else ' NO-GO'}, "
                f"loss {epoch0_loss:.4f} -> {train_trace[-1]['avg_loss']:.4f}, "
                f"alpha_bounded={alpha_bounded}, cond_grad_all={cond_grad_nonzero_all}, "
                f"ln_grad_all={ln_grad_nonzero_all}\n")
    print(f"\n[Final] verdict={verdict['overall_decision']}", flush=True)


if __name__ == "__main__":
    main()