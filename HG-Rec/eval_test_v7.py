"""v7 test eval: HG_Rec + Stage 2 codebook init for T5 SID embeddings → test.parquet beam=20."""
import os
import sys
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from data.dataset import GenRecDataset
from model.hg_rec_curv_v7 import HG_Rec_Curv_V7

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "./ckpt/Instruments/Aug-18-2026_v7/HG_Rec_curv_v7_epoch_0.pth"
STAGE2_CKPT = "./ckpt/Instruments/Aug-14-2026_20-04-16_beta_1.000_codebook_[64,128,256]_sk_0.000/best_collision_model.pth"
SID = "./dataset/Instruments/Instruments_t5_rqvae_beta_0.250_codebook_[64,128,256]_sk_0.000.npy"

config = dict(
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break
    return pos_index


def evaluate_test(model, loader, topk_list, beam_size, device):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    n_total = 0
    with torch.no_grad():
        for batch in tqdm(loader, ncols=100, desc="Test"):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    num_beams=beam_size)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            n_total += pos_index.shape[0]
            for k in topk_list:
                hits = pos_index[:, :k].any(dim=1).float().mean().item()
                ranks = torch.arange(1, k + 1).float()
                dcg_per_pos = 1.0 / torch.log2(ranks + 1)
                has_hit = pos_index[:, :k].any(dim=1)
                first_hit_idx = pos_index[:, :k].float().argmax(dim=1)
                dcg_at_hit = torch.where(has_hit, dcg_per_pos[first_hit_idx], torch.tensor(0.0))
                ndcg = dcg_at_hit.mean().item()
                recalls['Recall@' + str(k)].append(hits)
                ndcgs['NDCG@' + str(k)].append(ndcg)
    avg_recalls = {k: sum(v) / max(len(v), 1) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / max(len(v), 1) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs, n_total


def hgrec_collate_fn(batch, pad_token=0):
    histories = [item['history'] for item in batch]
    targets = [item['target'] for item in batch]
    flattened_histories = torch.stack(
        [torch.tensor([e for sub in h for e in sub], dtype=torch.int64) for h in histories]
    )
    flattened_targets = torch.stack(
        [torch.tensor(t, dtype=torch.int64) for t in targets]
    )
    attention_masks = torch.stack(
        [torch.tensor([1 if e != pad_token else 0 for e in h], dtype=torch.int64)
         for h in flattened_histories]
    )
    return {'history': flattened_histories, 'target': flattened_targets,
            'attention_mask': attention_masks}


def main():
    device = torch.device("cuda:0")
    model = HG_Rec_Curv_V7(config, stage2_ckpt_path=STAGE2_CKPT).to(device)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {CKPT}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:3]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:3]}")

    ds = GenRecDataset(
        dataset_path="./dataset/Instruments/test.parquet",
        code_path=SID, mode="evaluation",
        codebook_size=[64, 128, 256, 1], max_len=20)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=96, shuffle=False, num_workers=2,
                        collate_fn=hgrec_collate_fn)
    print("test samples:", len(ds))

    recalls, ndcgs, n = evaluate_test(model, loader, [5, 10, 20], 20, device)
    print(f"\n=== TEST (beam=20, N={n}) ===")
    for k in [5, 10, 20]:
        print(f"  Recall@{k}: {recalls['Recall@' + str(k)]:.4f}  NDCG@{k}: {ndcgs['NDCG@' + str(k)]:.4f}")
    import json
    out = {
        "ckpt": CKPT,
        "n_test": int(n),
        "metrics": {f"Recall@{k}": float(recalls[f'Recall@{k}']) for k in [5, 10, 20]},
        "ndcg": {f"NDCG@{k}": float(ndcgs[f'NDCG@{k}']) for k in [5, 10, 20]},
    }
    out_path = CKPT.replace(".pth", "_test.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()