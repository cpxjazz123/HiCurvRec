"""
Task #336 — Issue #43 Gate 2b Stage 4 — Test set evaluation.
Standalone eval loading best checkpoint, evaluating on test.parquet.
"""
import sys, os, json, glob, logging, importlib.util
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

# Import evaluate from training script fork
spec = importlib.util.spec_from_file_location(
    'train_fork',
    '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py',
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
evaluate = mod.evaluate

BASE = Path('/home/wlia0047/ar57/wenyu/GeneRec')

def main():
    # Config (mirror Stage 3)
    config = {
        'batch_size': 256, 'infer_size': 96, 'lr': 1e-4,
        'device': 'cuda:2',
        'num_layers': 6, 'num_decoder_layers': 4,
        'd_model': 128, 'd_ff': 1024, 'num_heads': 6, 'd_kv': 64,
        'dropout_rate': 0.1,
        'vocab_size': 1025, 'pad_token_id': 0, 'eos_token_id': 0,
        'feed_forward_proj': 'relu', 'max_len': 20,
        'dataset_name': 'Instruments',
        'dataset_path': str(BASE / 'HG-Rec' / 'dataset'),
        'codebook_size': [64, 128, 256, 1],
        'code_path': '_t5_hrqvae_hyp_pre.npy',
        'topk_list': [5, 10, 20],
        'beam_size': 20,
    }

    # Find best checkpoint
    ckpt_dir = BASE / 'products' / 'task336' / 'ckpt_hgrec' / 'Instruments'
    ckpt_list = sorted(ckpt_dir.glob('*/HG_Rec_best.pth'))
    if not ckpt_list:
        print("❌ No best checkpoint found")
        sys.exit(1)
    best_ckpt = str(ckpt_list[-1])
    print(f'[Stage 4] Loading best ckpt: {best_ckpt}')

    device = torch.device(config['device'])
    model = HG_Rec(config)
    model.load_state_dict(torch.load(best_ckpt, map_location='cpu'))
    model.to(device)
    model.eval()

    # Test dataset
    test_dataset = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'],
                               config['dataset_name'] + config['code_path']),
        mode='evaluation',
        codebook_size=config['codebook_size'],
        max_len=config['max_len']
    )
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
    print(f'[Stage 4] Test dataset size: {len(test_dataset)}')

    # Evaluate on TEST
    for beam_size in [20, 50]:
        print(f'\n===== Beam size = {beam_size} =====')
        config['beam_size'] = beam_size
        avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], beam_size, device)
        print(f'Test Recall: {avg_recalls}')
        print(f'Test NDCG: {avg_ndcgs}')

        result = {
            'best_ckpt': best_ckpt,
            'beam_size': beam_size,
            'test_recalls': avg_recalls,
            'test_ndcgs': avg_ndcgs,
        }
        result_json = BASE / f'verdicts/task336_issue43_gate2b_stage4_beam{beam_size}.json'
        with open(result_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'[Stage 4] Results saved: {result_json}')

    print('\n✅ Stage 4 complete for Issue #43 Gate 2b')

if __name__ == '__main__':
    main()
