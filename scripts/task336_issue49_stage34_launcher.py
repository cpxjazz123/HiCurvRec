"""
Task #336 — Issue #49 Stage 3+4 — T5-mini 训练 + Stage 4 评估.

复用 task84_hgrec_stage3_train.py 进行 T5-mini 200 epoch 训练,
完成后用 task174 模式做 Stage 4 评估.

Usage:
  python3 scripts/task336_issue49_stage34_launcher.py \
    --sid_npy products/task340/arm_plus/sid.npy \
    --arm_name arm_plus \
    --device cuda:0
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sid_npy', required=True, help='Stage 2 SID output')
    p.add_argument('--arm_name', required=True, help='arm name for paths')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num_epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    return p.parse_args()


def main():
    args = parse_args()
    base = Path('/home/wlia0047/ar57/wenyu/GeneRec')

    # ── Stage 3: T5-mini 训练 ──
    stage3_out = base / f'products/task340/{args.arm_name}/t5_out'
    stage3_log = base / f'logs/task340_st3_{args.arm_name}.log'
    stage3_out.mkdir(parents=True, exist_ok=True)
    os.makedirs(base / 'logs', exist_ok=True)

    # SID file needs to be in HG-Rec/dataset/Instruments/ for task84 script
    code_path = f'_t5_rqvae_{args.arm_name}.npy'
    sid_target = base / f'HG-Rec/dataset/Instruments/{code_path}'
    sid_target.parent.mkdir(parents=True, exist_ok=True)

    # Copy SID npy to where T5 expects it
    import shutil
    shutil.copy2(args.sid_npy, sid_target)
    print(f"SID copied to {sid_target}")

    cmd = [
        str(base / 'HG-Rec' / 'scripts' / 'task84_hgrec_stage3_train.py'),
        '--dataset_name', 'Instruments',
        '--dataset_path', str(base / 'HG-Rec' / 'dataset'),
        '--code_path', code_path,
        '--codebook_size', '64', '128', '256', '1',
        '--num_epochs', str(args.num_epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--num_layers', '6',
        '--num_decoder_layers', '4',
        '--d_model', '128',
        '--d_ff', '1024',
        '--num_heads', '6',
        '--d_kv', '64',
        '--vocab_size', '1025',
        '--max_len', '20',
        '--pad_token_id', '0',
        '--eos_token_id', '0',
        '--device', args.device,
        '--mode', 'train',
        '--save_path', str(stage3_out),
        '--log_path', str(base / 'logs'),
        '--seed', '42',
        '--early_stop', '20',
        '--beam_size', '20',
        '--infer_size', '96',
    ]

    print(f"\nStage 3 T5-mini training ({args.num_epochs} epochs)...")
    print(f"  Device: {args.device}")
    print(f"  SID: {args.sid_npy}")
    print(f"  Log: {stage3_log}")

    env = os.environ.copy()
    env['PYTHONPATH'] = str(base / 'HG-Rec') + ':' + env.get('PYTHONPATH', '')
    env['HF_HOME'] = '/home/wlia0047/ar57_scratch/wenyu/hf_models'
    env['TRITON_CACHE_DIR'] = f'/home/wlia0047/.triton/cache_task336_st3_{args.arm_name}'
    os.makedirs(env['TRITON_CACHE_DIR'], exist_ok=True)

    t_start = time.time()
    with open(stage3_log, 'w') as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(base / 'HG-Rec'))
    t_elapsed = time.time() - t_start

    if result.returncode != 0:
        print(f"❌ Stage 3 failed (rc={result.returncode}), check {stage3_log}")
        sys.exit(1)

    print(f"✅ Stage 3 complete in {t_elapsed/60:.1f} min")

    # ── Stage 4: 评估 (beam=20/50) ──
    print(f"\nStage 4 evaluation...")
    # Find the best checkpoint
    import glob
    ckpt_dirs = list(stage3_out.glob('Instruments/*/'))
    if not ckpt_dirs:
        print("❌ No checkpoint directory found")
        sys.exit(1)
    best_ckpt_dir = sorted(ckpt_dirs)[-1]
    best_ckpt = best_ckpt_dir / 'HG_Rec_best.pth'
    if not best_ckpt.exists():
        print(f"❌ No best checkpoint at {best_ckpt}")
        sys.exit(1)
    print(f"  Checkpoint: {best_ckpt}")

    # Run eval with beam=20 (reuse task84_hgrec_stage3_train.py in eval mode)
    for beam_size in [20, 50]:
        print(f"\n  Beam size = {beam_size}")
        eval_log = base / f'logs/task340_eval_{args.arm_name}_beam{beam_size}.log'
        eval_cmd = cmd.copy()
        eval_cmd[cmd.index('--mode') + 1] = 'evaluation'
        # Replace beam size
        eval_cmd[cmd.index('--beam_size') + 1] = str(beam_size)
        # Load checkpoint
        eval_cmd.extend(['--resume_path', str(best_ckpt)])

        with open(eval_log, 'w') as f:
            result = subprocess.run(eval_cmd, env=env, stdout=f,
                                    stderr=subprocess.STDOUT,
                                    cwd=str(base / 'HG-Rec'))
        if result.returncode != 0:
            print(f"  ⚠️ Eval beam={beam_size} rc={result.returncode}")
        else:
            print(f"  ✅ beam={beam_size} done")

    print(f"\n✅ Stage 3+4 complete for {args.arm_name}")
    print(f"Results in: {stage3_log}")


if __name__ == "__main__":
    main()
