"""Issue #165: #162 Treatment 内部消融归因 (同一 checkpoint 一次性干预).

分支:
  T_full      = confidence gate + layer mixer
  T_no_mixer  = confidence gate + uniform layer weight (a=1/3 同尺度)
  T_no_gate   = layer mixer + g=1
  T_off       = gate/mixer/curvature injection 全部关闭
  layerwise   = 只 L0 / 只 L1 / 只 L2 / layer-weight shuffle

固定输入: #162 Treatment stage2 ckpt + SID + curvature_state, stage3 ckpt.
同一批 24,772 Test 样本, 同一 beam=20.
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TASK_DIR = Path(__file__).parent
SRC = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue162_layer_curvature_mixer/treatment")
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(TASK_DIR / "_lib"))  # history_s 版 curvature_dataset 优先

from HG_Rec import HG_Rec
from dataset import GenRecDataset
from curvature_dataset import CurvatureGenRecDataset

# 从 #162 stage3 复用 injector 类
import importlib.util
spec = importlib.util.spec_from_file_location("stage3_src", str(SRC / "stage3.py"))
stage3_src = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage3_src)
CurvatureStateInjector = stage3_src.CurvatureStateInjector
install_curvature_state = stage3_src.install_curvature_state

# ── 配置 (与 #162 treatment 一致) ──
SEED = 42
BATCH_SIZE = 96
BEAM_SIZE = 20
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
EVAL_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/test.parquet"
SID_NPY = str(SRC / "stage2" / "sid_output.npy")
CURV_STATE = str(SRC / "stage2" / "curvature_state.npy")
CKPT = str(SRC / "stage2" / "HG_Rec_best.pth")
STAGE2_CKPT = str(SRC / "stage2" / "hrqvae_kappa_sync.ckpt")
CONFIG = dict(
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)
_LAYER_ID_LUT = np.full(1025, -1, dtype=np.int64)
_LAYER_ID_LUT[1:65] = 0
_LAYER_ID_LUT[65:193] = 1
_LAYER_ID_LUT[193:449] = 2
_LAYER_ID_LUT[449:450] = 3


def _curv_collate(batch, pad_token=0):
    histories = [item['history'] for item in batch]
    targets = [item['target'] for item in batch]
    flattened_histories = torch.stack(
        [torch.tensor([elem for sublist in h for elem in sublist], dtype=torch.int64) for h in histories])
    flattened_targets = torch.stack([torch.tensor(t, dtype=torch.int64) for t in targets])
    attention_masks = torch.stack(
        [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64)
         for h in flattened_histories])
    out = {'history': flattened_histories, 'target': flattened_targets, 'attention_mask': attention_masks}
    if 'history_c' in batch[0]:
        flat_c = torch.stack([torch.tensor([e for sl in it['history_c'] for e in sl], dtype=torch.float32)
                              for it in batch])
        out['history_c'] = flat_c
    if 'history_s' in batch[0]:
        flat_s = torch.stack([torch.tensor([e for sl in it['history_s'] for e in sl], dtype=torch.float32)
                              for it in batch])
        out['history_s'] = flat_s.view(flat_s.shape[0], -1, 4)
    return out


class ModeInjector(CurvatureStateInjector):
    """#165: 直接继承 #162 injector (参数 key 完全一致), 加 mode 干预开关."""
    def __init__(self, d_model, feat_dim=4, hidden=16, alpha_init=0.0, tau=1.0, n_layers=3):
        super().__init__(d_model, feat_dim=feat_dim, hidden=hidden,
                         alpha_init=alpha_init, tau=tau, n_layers=n_layers)
        self.mode = "full"
        self._gate_calls = 0
        self._mixer_calls = 0

    def forward(self, curvature, response, layer_ids=None):
        b = self
        self._gate_calls += 1
        g = torch.sigmoid(b.gate_mlp(response))  # (B, L, 1)
        if self.mode == "off":
            return torch.zeros(curvature.shape[0], curvature.shape[1], b.proj.out_features,
                               device=curvature.device)
        if self.mode == "no_gate":
            g = torch.ones_like(g)
        if layer_ids is not None:
            self._mixer_calls += 1
            c_feat = response[:, :, :3].mean(dim=1)
            q_pool = torch.full((response.shape[0], 2), 0.5, device=response.device)  # 与 #162 一致
            m_in = torch.cat([q_pool, c_feat], dim=-1)
            a = torch.softmax(b.mixer(m_in) / b.tau, dim=-1)  # (B, 3)
            if self.mode == "no_mixer":
                a = torch.full_like(a, 1.0 / 3.0)
            elif self.mode == "shuffle":
                perm = torch.randperm(3, device=a.device)
                a = a[:, perm]
            elif self.mode in ("l0_only", "l1_only", "l2_only"):
                keep = int(self.mode[1])
                a = torch.zeros_like(a).scatter(1, torch.full((a.shape[0], 1), keep, device=a.device, dtype=torch.long), 1.0)
            a_t = a.gather(1, layer_ids.clamp(0, 2))  # (B, L)
        else:
            a_t = torch.ones_like(curvature)
        delta = b.alpha * a_t.unsqueeze(-1) * g * b.proj(curvature.unsqueeze(-1))
        return delta


def install_mode_injector(hg_rec, mode_injector):
    device = next(hg_rec.parameters()).device
    mode_injector = mode_injector.to(device)
    d_model_sqrt = hg_rec.model.config.d_model ** 0.5
    hg_rec.add_module("curv_injector", mode_injector)

    def curv_forward(self, input_ids, attention_mask=None, labels=None, curvature=None, response=None):
        input_embeds = self.model.shared(input_ids) * d_model_sqrt
        if curvature is not None and curvature.shape[1] == input_ids.shape[1]:
            if response is not None:
                layer_ids = torch.from_numpy(_LAYER_ID_LUT[input_ids.cpu().numpy()].clip(0, 2)).to(input_ids.device)
                input_embeds = input_embeds + self.curv_injector(curvature, response, layer_ids)
            else:
                input_embeds = input_embeds + self.curv_injector(
                    curvature, torch.zeros(curvature.shape[0], curvature.shape[1], 4, device=curvature.device))
        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

    def curv_generate(self, input_ids, attention_mask=None, num_beams=20, **kwargs):
        curvature = kwargs.pop("curvature", None)
        response = kwargs.pop("response", None)
        input_embeds = self.model.shared(input_ids) * d_model_sqrt
        if curvature is not None and curvature.shape[1] == input_ids.shape[1]:
            if response is not None:
                layer_ids = torch.from_numpy(_LAYER_ID_LUT[input_ids.cpu().numpy()].clip(0, 2)).to(input_ids.device)
                input_embeds = input_embeds + self.curv_injector(curvature, response, layer_ids)
            else:
                input_embeds = input_embeds + self.curv_injector(
                    curvature, torch.zeros(curvature.shape[0], curvature.shape[1], 4, device=curvature.device))
        return self.model.generate(inputs_embeds=input_embeds, attention_mask=attention_mask,
                                   num_beams=num_beams, max_length=5, num_return_sequences=num_beams, **kwargs)

    import types
    hg_rec.forward = types.MethodType(curv_forward, hg_rec)
    hg_rec.generate = types.MethodType(curv_generate, hg_rec)
    return hg_rec


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).float()


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert preds.shape[1] == maxk
    pos_index = (preds == labels.unsqueeze(1)).all(dim=-1)
    return pos_index


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_branch(model, injector, loader, device, mode, out_dir):
    injector.mode = mode
    injector._gate_calls = 0
    injector._mixer_calls = 0
    model.eval()
    print(f'[debug] {mode}: generate_type={type(model.generate).__name__} has_inj={hasattr(model, "curv_injector")}')
    recalls = {f"R@{k}": [] for k in [5, 10, 20]}
    ndcgs = {f"NDCG@{k}": [] for k in [5, 10, 20]}
    raw_rows = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            curv = batch["history_c"].to(device) if "history_c" in batch else None
            resp = batch["history_s"].to(device) if "history_s" in batch else None
            if curv is not None and resp is not None:
                preds = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       num_beams=BEAM_SIZE, curvature=curv, response=resp)
            else:
                preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=BEAM_SIZE)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], BEAM_SIZE, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=BEAM_SIZE)
            for k in [5, 10, 20]:
                recalls[f"R@{k}"].append(recall_at_k(pos_index, k).sum().item())
                ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k).sum().item())
            for s in range(input_ids.shape[0]):
                pi = pos_index[s].cpu().tolist()
                best_rank = next((r + 1 for r, v in enumerate(pi) if v), 21)
                raw_rows.append({
                    "sample_idx": bi * BATCH_SIZE + s,
                    "hit_at_5": int(best_rank <= 5),
                    "hit_at_10": int(best_rank <= 10),
                    "hit_at_20": int(best_rank <= 20),
                    "best_rank_top20": best_rank,
                })
    n = len(loader.dataset)
    result = {k: sum(v) / n for k, v in recalls.items()}
    result.update({k: sum(v) / n for k, v in ndcgs.items()})
    result["n_eval"] = n
    result["gate_calls"] = injector._gate_calls
    result["mixer_calls"] = injector._mixer_calls
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_test.json", "w") as f:
        json.dump(result, f, indent=2)
    import pandas as pd
    pd.DataFrame(raw_rows).to_parquet(out_dir / "raw_predictions_full.parquet", index=False)
    print(f"[{mode}] R@5/10/20 = {result['R@5']:.4f}/{result['R@10']:.4f}/{result['R@20']:.4f} "
          f"NDCG@10 = {result['NDCG@10']:.4f} gate_calls={injector._gate_calls} mixer_calls={injector._mixer_calls}")
    return result


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)

    # ── 固定输入 manifest ──
    manifest = {
        "stage2_ckpt_sha": sha256_file(STAGE2_CKPT),
        "sid_sha": sha256_file(SID_NPY),
        "curv_state_sha": sha256_file(CURV_STATE),
        "stage3_ckpt_sha": sha256_file(CKPT),
        "source": {
            "stage3": sha256_file(str(SRC / "stage3.py")),
            "stage4": sha256_file(str(SRC / "stage4.py")),
            "cross_layer_quantizer": sha256_file(str(SRC / "_lib" / "cross_layer_quantizer.py")),
            "curvature_dataset": sha256_file(str(SRC / "_lib" / "curvature_dataset.py")),
        },
    }
    with open(TASK_DIR / "source_and_checkpoint_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("[165] manifest saved (stage2/3 ckpt + source hashes)")

    # ── dataset + loader ──
    ds = CurvatureGenRecDataset(
        dataset_path=EVAL_PARQUET, code_path=SID_NPY, curvature_path=CURV_STATE,
        mode="evaluation", codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=0, collate_fn=_curv_collate)

    # ── 模型 + injector ──
    model = HG_Rec(CONFIG).to(device)
    mode_inj = ModeInjector(d_model=CONFIG["d_model"])
    model = install_mode_injector(model, mode_inj)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[165] ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)

    # ── 4 分支 + layerwise ──
    branches = {
        "full": "full",
        "no_mixer": "no_mixer",
        "no_gate": "no_gate",
        "off": "off",
        "l0_only": "l0_only",
        "l1_only": "l1_only",
        "l2_only": "l2_only",
        "shuffle": "shuffle",
    }
    results = {}
    for name, mode in branches.items():
        results[name] = evaluate_branch(model, mode_inj, loader, device, mode,
                                        TASK_DIR / name if name in ("full", "no_mixer", "no_gate", "off")
                                        else TASK_DIR / "layerwise_ablation" / name)

    with open(TASK_DIR / "module_contribution.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[165] all branches done -> module_contribution.json")


if __name__ == "__main__":
    main()
