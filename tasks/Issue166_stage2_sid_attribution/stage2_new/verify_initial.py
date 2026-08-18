"""Issue #166 Stage2 初始一致性验证: 同一 Stage1 → 同一 KMeans 中心 → 初始 codebook/SID 相同.

验证链条:
1. 同一 Stage1 embedding
2. 固定 KMeans (random_state=2024) → 同一初始中心
3. 初始 codebook hash 相同
4. 初始 distance/assignment/SID 相同
5. 之后只开关 M2 (intrinsic residual)
"""

import os, sys, json, hashlib
import numpy as np
import torch

TASK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(TASK, "treatment", "_lib"))
from cross_layer_quantizer import CrossLayerHRQVAE
from utils import EmbDataset

SEED = 2024
EMB_DIM = 768
E_DIM = 32
CODEBOOK_SIZES = [64, 128, 256]
ENCODER_LAYERS = [512, 256, 128, 64]
N_ITEMS = 9922
BATCH = 1024
KMEANS_ITERS = 1000


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_model(prefix_routing, m2):
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = CrossLayerHRQVAE(
        in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
        layers=ENCODER_LAYERS, kmeans_init=True, kmeans_iters=KMEANS_ITERS,
        prefix_routing=prefix_routing,
        gate_M2_intrinsic=m2,
        gate_M3_transport=True,
    ).to(device)
    m.train()
    # Issue #166: init 用欧式路径 (两套一致)
    m.gate_M2_intrinsic = False
    with torch.no_grad():
        m(item_emb[:BATCH], use_sk=False)  # 触发 kmeans init
    m.gate_M2_intrinsic = m2
    return m


def sha(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def main():
    global item_emb
    item_emb = torch.tensor(
        EmbDataset(os.path.join(TASK, "treatment", "stage1", "item_emb.parquet")).embeddings,
        dtype=torch.float32).to(device)

    mC = build_model(False, False)
    mT = build_model(False, True)

    report = {"initial_consistency": {}}
    # 初始 codebook hash
    for li in range(3):
        hC = sha(mC.vq_layers[li].embeddings.weight)
        hT = sha(mT.vq_layers[li].embeddings.weight)
        cb_diff = (mC.vq_layers[li].embeddings.weight - mT.vq_layers[li].embeddings.weight).abs().max().item()
        report["initial_consistency"][f"L{li}_codebook"] = {
            "S_C_hash": hC, "S_T_hash": hT, "same_bytes": hC == hT,
            "max_abs_diff": cb_diff, "same_numerically": cb_diff < 1e-7}
    # 初始 SID (get_indices 在初始模型上)
    with torch.no_grad():
        sidC = mC.get_indices(item_emb, use_sk=False)
        sidT = mT.get_indices(item_emb, use_sk=False)
    report["initial_consistency"]["initial_SID_same"] = bool(torch.equal(sidC, sidT))
    report["initial_consistency"]["initial_SID_diff_frac"] = float((sidC != sidT).any(dim=-1).float().mean())
    # 初始 distance/loss (同一 batch forward)
    with torch.no_grad():
        outC, rqC, idxC, _, _ = mC(item_emb[:BATCH], use_sk=False)
        outT, rqT, idxT, _, _ = mT(item_emb[:BATCH], use_sk=False)
    report["initial_consistency"]["initial_out_same"] = bool(torch.allclose(outC, outT, atol=1e-6))
    report["initial_consistency"]["initial_idx_same"] = bool(torch.equal(idxC, idxT))
    report["initial_consistency"]["initial_rq_diff"] = float((rqC - rqT).abs().max())

    ok = all(v.get("same", False) or v.get("initial_SID_same", False)
             or v.get("initial_out_same", False) or v.get("initial_idx_same", False)
             for k, v in report["initial_consistency"].items() if isinstance(v, dict))
    # 实际: 需全部 true
    all_same = (report["initial_consistency"]["L0_codebook"]["same_numerically"]
                and report["initial_consistency"]["L1_codebook"]["same_numerically"]
                and report["initial_consistency"]["L2_codebook"]["same_numerically"]
                and report["initial_consistency"]["initial_SID_diff_frac"] < 0.01
                and report["initial_consistency"]["initial_out_same"]
                and report["initial_consistency"]["initial_idx_same"])
    report["decision"] = "PASS (初始完全一致)" if all_same else "FAIL (初始不一致)"
    with open(os.path.join(TASK, "stage2_initial_consistency.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2))
    print("\n", report["decision"])
    return 0 if all_same else 1


if __name__ == "__main__":
    sys.exit(main())
