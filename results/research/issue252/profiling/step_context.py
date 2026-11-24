"""Shared construction of the real candidate training context (diagnostic only)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
TREE = ROOT / 'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq'
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

_spec = importlib.util.spec_from_file_location('cand_train', TREE / 'train_rqvae.py')
cand = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cand)


def build(device_index=0):
    """Model, loader, optimizer, catalogue and sampler exactly as the entry builds them."""
    device = torch.device('cuda:%d' % device_index)
    torch.manual_seed(cand.SEED)
    embeddings = cand.maybe_apply_pca(cand.load_embeddings(cand.EMBEDDING_FILE),
                                      cand.PCA_DIM, cand.SEED)
    frame = cand.pd.read_parquet(cand.TRAIN_FILE)
    train_ids = np.unique(frame['target'].to_numpy(dtype=np.int64))
    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
    config = cand._tokenizer_config()
    model = cand.RQVAE(config, in_dim=embeddings.shape[1], distance=cand.DISTANCE).to(device)
    if cand.XAVIER_INIT:
        cand.initialize_tiger_weights(model)
    dataset = cand._BatchedTensorDataset(train_embeddings, torch.from_numpy(train_ids))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cand.BATCH_SIZE_PER_RANK, shuffle=True, num_workers=cand.NUM_WORKERS,
        collate_fn=cand._batch_passthrough, pin_memory=True, persistent_workers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cand.LR, weight_decay=cand.WEIGHT_DECAY)
    catalogue = all_embeddings.to(device)
    with torch.no_grad():
        model.init_codebook(train_embeddings.to(device))
    model.freeze_tangent_scale(catalogue)
    sampler = model.structural_sampler()
    calibration_ids = torch.from_numpy(train_ids[:max(cand.CODEBOOK_SIZE)]).to(device)
    model.calibrate_curvature_weight(
        catalogue[calibration_ids], calibration_ids, catalogue,
        structural_sample=sampler.draw(0, 0, 1, track=False))
    return {'model': model, 'loader': loader, 'optimizer': optimizer, 'catalogue': catalogue,
            'sampler': sampler, 'device': device, 'train_ids': train_ids, 'cand': cand}
