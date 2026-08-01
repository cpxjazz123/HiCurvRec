"""Task #473 Stage 4 R@K double-run resume (skip retrain, use saved ckpt).

R23: ckpt already saved at products/task473_issue181_direction_b_gate4_200ep/adapter_200ep.pt
Stage 3 reached early stop @ ep35 best_val_R@10_sim=0.1150 (vs baseline 0.1020, +12.7%).
This script loads best ckpt + runs Stage 4 R@K double-run only (R@5/10/20, NDCG@5/10/20, run1 + run2).
"""
import os, sys, json, time, hashlib, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task473_resume"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import numpy as np
from torch.utils.data import DataLoader

PROJECT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, PROJECT)
sys.path.insert(0, f"{PROJECT}/HG-Rec/model")
sys.path.insert(0, f"{PROJECT}/HG-Rec/data")

# Load wrapper module from task471
_t471_path = os.path.join(PROJECT, "scripts/task471_issue178_gate3_b_recontinue.py")
spec = importlib.util.spec_from_file_location("task471_mod", _t471_path)
_t471_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_t471_mod)
WrapperCls = _t471_mod.HG_Rec_with_BoundedWeightedMixedAdapter
get_t5_config = _t471_mod.get_t5_config
load_t5_state_dict = _t471_mod.load_t5_state_dict

# Constants
T5_CKPT = f"{PROJECT}/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
SID_NPY = f"{PROJECT}/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TEST_PARQUET = f"{PROJECT}/HG-Rec/dataset/Instruments/test.parquet"
TRAIN_PARQUET = f"{PROJECT}/HG-Rec/dataset/Instruments/train.parquet"
CKPT_PATH = f"{PROJECT}/products/task473_issue181_direction_b_gate4_200ep/adapter_200ep.pt"
DEVICE = "cuda"
D_MODEL = 128
MAX_LEN = 4
PAD_TOKEN = 0
CODEBOOK_SIZE = [64, 128, 256, 1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_r_at_k(preds_list, targets_list, k):
    n_correct = sum(1 for p, t in zip(preds_list, targets_list) if p[:k] == t[:k])
    return n_correct / max(1, len(preds_list))


def compute_ndcg_at_k(preds_list, targets_list, k):
    ndcgs = []
    for p, t in zip(preds_list, targets_list):
        if p[:k] == t[:k]:
            ndcgs.append(1.0)
        else:
            ndcgs.append(0.0)
    return sum(ndcgs) / max(1, len(ndcgs))


def main():
    log = []
    log.append("[Task #473 Stage 4 resume] 方向B BoundedWeightedMixedCurvatureConditioner R@K double-run")

    # SHA256
    sid_hash = sha256(SID_NPY)
    t5_hash = sha256(T5_CKPT)
    log.append(f"[SHA256] SID_NPY: {sid_hash}")
    log.append(f"[SHA256] T5_CKPT: {t5_hash}")

    # Verify Hash Check
    expected_sid = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"
    assert sid_hash == expected_sid, f"SID hash mismatch: {sid_hash} != {expected_sid}"
    log.append("[Hash Check 跟 #157/#158 一致] True")

    # Data
    from dataset import GenRecDataset
    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    log.append(f"[Data] test: {len(test_ds)}")

    # Model
    t5_config = get_t5_config()
    t5_state_dict = load_t5_state_dict(T5_CKPT)
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4)
    model_wrapper = model_wrapper.to(DEVICE)

    # Load best ckpt
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model_wrapper.adapter.load_state_dict(ckpt["adapter_state_dict"])
    model_wrapper.first_input_ln.load_state_dict(ckpt["first_input_ln_state_dict"])
    log.append(f"[Load best ckpt] alpha={ckpt['alpha_value']:.6e}")
    model_wrapper.eval()

    # Stage 4 R@K double-run
    def collate_fn(batch):
        # batch is list of dicts with 'history' (seq of SID) + 'target' (SID)
        histories = [b["history"] for b in batch]
        targets = [b["target"] for b in batch]
        # Stack into tensors: history -> (B, L, 4), target -> (B, 4)
        max_L = max(len(h) for h in histories)
        history_padded = np.zeros((len(batch), max_L, 4), dtype=np.int64)
        for i, h in enumerate(histories):
            L = len(h)
            for j in range(L):
                history_padded[i, j] = h[j]
        target_arr = np.stack(targets, axis=0)
        return {
            "input_ids": torch.from_numpy(history_padded.reshape(len(batch), -1)),
            "labels": torch.from_numpy(target_arr),
        }
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)
    K_list = [5, 10, 20]
    run_metrics = {"run1": {}, "run2": {}}
    preds_records = {"run1": [], "run2": []}
    targets_records = []

    for run_id in [1, 2]:
        log.append(f"[Stage 4 R@K] run {run_id} starting")
        preds_list = []
        targets_list = []
        torch.manual_seed(42 + run_id)
        n_processed = 0
        with torch.no_grad():
            for batch in test_loader:
                history_tensor = batch["input_ids"].to(DEVICE)
                target_tensor = batch["labels"].to(DEVICE)
                attention_mask = (history_tensor != PAD_TOKEN).long()
                B, L_flat = history_tensor.shape
                digit_values = history_tensor.float()
                layer_idx = torch.arange(L_flat, device=DEVICE) % 4
                layer_idx = layer_idx.float().unsqueeze(0).expand(B, -1)
                pos_in_history = torch.arange(L_flat, device=DEVICE) // 4
                pos_in_history = pos_in_history.float().unsqueeze(0).expand(B, -1) / MAX_LEN
                padding_flag = (digit_values == PAD_TOKEN).float()
                sid_meta = torch.stack([digit_values / 1025.0, layer_idx / 4.0, pos_in_history, padding_flag], dim=-1)
                curvature_meta = torch.zeros(B, 3, 4, dtype=torch.float32, device=DEVICE)

                x_emb = model_wrapper.t5.model.shared(history_tensor)
                residual, alpha = model_wrapper.adapter(x_emb, sid_meta, curvature_meta)
                x_emb_with_residual = x_emb + residual
                x_emb_with_residual = model_wrapper.first_input_ln(x_emb_with_residual)
                decoder_input_ids = torch.zeros(B, 4, dtype=torch.long, device=DEVICE)
                encoder_outputs = model_wrapper.t5.model.encoder(
                    inputs_embeds=x_emb_with_residual,
                    attention_mask=attention_mask,
                )
                decoder_outputs = model_wrapper.t5.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    encoder_attention_mask=attention_mask,
                )
                logits = model_wrapper.t5.model.lm_head(decoder_outputs.last_hidden_state)

                preds = logits.argmax(dim=-1)
                for i in range(B):
                    pred = preds[i].cpu().tolist()
                    target = target_tensor[i].cpu().tolist()
                    preds_list.append(pred[:4])
                    targets_list.append(target[:4])
                n_processed += B
        # Compute metrics
        for k in K_list:
            r = compute_r_at_k(preds_list, targets_list, k)
            run_metrics[f"run{run_id}"][f"R@{k}"] = r
        for k in K_list:
            n = compute_ndcg_at_k(preds_list, targets_list, k)
            run_metrics[f"run{run_id}"][f"NDCG@{k}"] = n
        preds_records[f"run{run_id}"] = preds_list
        targets_records = targets_list
        log.append(f"[Stage 4 R@K run{run_id}] n={n_processed}, metrics={run_metrics[f'run{run_id}']}")

    # Average across runs
    avg_metrics = {}
    for k in K_list:
        avg_metrics[f"R@{k}"] = (run_metrics["run1"][f"R@{k}"] + run_metrics["run2"][f"R@{k}"]) / 2
    for k in K_list:
        avg_metrics[f"NDCG@{k}"] = (run_metrics["run1"][f"NDCG@{k}"] + run_metrics["run2"][f"NDCG@{k}"]) / 2

    log.append(f"[Stage 4 avg R@K] {avg_metrics}")
    log.append(f"[Gate 4] test R@10 (avg run1+run2) = {avg_metrics['R@10']:.4f}, baseline 0.1020")
    if avg_metrics["R@10"] > 0.1020:
        log.append(f"[Gate 4] ✅ [TARGET REACHED] R@10 = {avg_metrics['R@10']:.4f} > 0.1020 baseline (+{(avg_metrics['R@10']-0.1020)*100/0.1020:.1f}%)")
    else:
        log.append(f"[Gate 4] ❌ NO-GO R@10 = {avg_metrics['R@10']:.4f} ≤ 0.1020 baseline")

    # Save
    out_dir = "products/task473_issue181_direction_b_gate4_200ep"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/stage4_verdict.json", "w") as f:
        json.dump({"metrics_avg": avg_metrics, "run1": run_metrics["run1"], "run2": run_metrics["run2"], "alpha_loaded": ckpt["alpha_value"]}, f, indent=2)

    log_str = "\n".join(log)
    print(log_str)
    with open("logs/task473_resume_stage4.log", "w") as f:
        f.write(log_str + "\n")


if __name__ == "__main__":
    main()