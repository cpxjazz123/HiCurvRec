"""
Task #339 / Task #340 — Issue #52 / Issue #53 Stage 3 + 4 统一 launcher.

Stage 3: T5-mini 训练 (跟 #49 baseline recipe 一致, 200 epoch, early stop 20)
Stage 4: test eval (beam=20) → 写 verdicts/<task>_stage4_beam20.json

用法:
  # Issue #52 Arm A
  python3 scripts/task339_issue52_stage34_launcher.py \\
    --sid_npy products/task339/arm_a/sid.npy \\
    --task_id 339 --arm_name arm_a \\
    --device cuda:0

  # Issue #53 Arm B
  python3 scripts/task339_issue52_stage34_launcher.py \\
    --sid_npy products/task340/arm_b/sid.npy \\
    --task_id 340 --arm_name arm_b \\
    --device cuda:0
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time, json
from pathlib import Path

BASE = Path('/home/wlia0047/ar57/wenyu/GeneRec')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sid_npy', required=True,
                   help='Path to 4-digit SID .npy from Stage 2')
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--arm_name', required=True,
                   help='Arm identifier for output paths (e.g. arm_a, arm_b)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num_epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--early_stop', type=int, default=20)
    p.add_argument('--beam_size', type=int, default=20)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--num_decoder_layers', type=int, default=4)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_ff', type=int, default=1024)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--d_kv', type=int, default=64)
    p.add_argument('--vocab_size', type=int, default=1025)
    p.add_argument('--max_len', type=int, default=20)
    p.add_argument('--infer_size', type=int, default=96)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--skip_stage3', action='store_true',
                   help='Skip Stage 3 (use existing ckpt)')
    p.add_argument('--skip_stage4', action='store_true',
                   help='Skip Stage 4 eval (only Stage 3)')
    return p.parse_args()


def main():
    args = parse_args()

    sid_name = Path(args.sid_npy).stem  # e.g. sid
    # task84 script builds code_path as <dataset_name> + code_path
    # i.e. <dataset_path>/Instruments/Instruments<code_path>.npy
    # So if code_path = "_t5_rqvae_task339_arm_a", final file = "Instruments_t5_rqvae_task339_arm_a.npy"
    target_sid = BASE / 'HG-Rec' / 'dataset' / 'Instruments' / f'Instruments_t5_rqvae_task{args.task_id}_{args.arm_name}.npy'
    target_sid.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    if not target_sid.exists():
        shutil.copy2(args.sid_npy, target_sid)
        print(f"SID copied: {args.sid_npy} → {target_sid}")
    else:
        print(f"SID already at target: {target_sid}")

    code_path_arg = f'_t5_rqvae_task{args.task_id}_{args.arm_name}.npy'

    out_dir = BASE / f'products/task{args.task_id}/{args.arm_name}/t5_out'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = BASE / 'logs'
    log_dir.mkdir(exist_ok=True)

    stage3_log = log_dir / f'task{args.task_id}_{args.arm_name}_st3.log'
    stage4_log = log_dir / f'task{args.task_id}_{args.arm_name}_st4.log'

    ckpt_path = out_dir / 'best_ckpt.pth'  # Stage 3 saves via task84 script

    # ─── Stage 3: T5-mini training ───
    if not args.skip_stage3:
        if ckpt_path.exists() or (out_dir / 'best_loss_model.pth').exists():
            print(f"⚠️ Stage 3 ckpt already exists at {out_dir}, skipping (use --skip_stage3 flag to bypass)")
        else:
            cmd = [
                '/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3',
                str(BASE / 'scripts/task84_hgrec_stage3_train.py'),
                '--dataset_name', 'Instruments',
                '--dataset_path', str(BASE / 'HG-Rec/dataset'),
                '--code_path', code_path_arg,
                '--codebook_size', '64', '128', '256', '1',
                '--num_epochs', str(args.num_epochs),
                '--batch_size', str(args.batch_size),
                '--lr', str(args.lr),
                '--num_layers', str(args.num_layers),
                '--num_decoder_layers', str(args.num_decoder_layers),
                '--d_model', str(args.d_model),
                '--d_ff', str(args.d_ff),
                '--num_heads', str(args.num_heads),
                '--d_kv', str(args.d_kv),
                '--vocab_size', str(args.vocab_size),
                '--max_len', str(args.max_len),
                '--pad_token_id', '0',
                '--eos_token_id', '0',
                '--device', args.device,
                '--mode', 'train',
                '--save_path', str(out_dir),
                '--log_path', str(log_dir),
                '--seed', str(args.seed),
                '--early_stop', str(args.early_stop),
                '--beam_size', str(args.beam_size),
                '--infer_size', str(args.infer_size),
            ]
            env = os.environ.copy()
            env['PYTHONPATH'] = str(BASE / 'HG-Rec') + ':' + env.get('PYTHONPATH', '')
            env['HF_HOME'] = '/home/wlia0047/ar57_scratch/wenyu/hf_models'
            env['TRITON_CACHE_DIR'] = f'/home/wlia0047/.triton/cache_task{args.task_id}_{args.arm_name}_st3'
            os.makedirs(env['TRITON_CACHE_DIR'], exist_ok=True)

            print(f"\nStage 3 T5-mini training ({args.num_epochs} epochs)...")
            print(f"  Device: {args.device}")
            print(f"  Log: {stage3_log}")
            print(f"  Output: {out_dir}")

            t0 = time.time()
            with open(stage3_log, 'w') as f:
                result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                        cwd=str(BASE))
            dt = time.time() - t0
            if result.returncode != 0:
                print(f"❌ Stage 3 failed (rc={result.returncode}), check {stage3_log}")
                sys.exit(1)
            print(f"✅ Stage 3 complete in {dt/60:.1f} min")

    # ─── Stage 4: test eval beam=20 ───
    if args.skip_stage4:
        print("Stage 4 skipped by --skip_stage4")
        return

    # Find best ckpt
    candidate_ckpts = [out_dir / 'best_ckpt.pth', out_dir / 'best_loss_model.pth', out_dir / 'best_collision_model.pth']
    best_ckpt = None
    for c in candidate_ckpts:
        if c.exists():
            best_ckpt = c
            break
    if best_ckpt is None:
        print(f"❌ No ckpt found in {out_dir}, abort Stage 4")
        sys.exit(1)
    print(f"Using ckpt: {best_ckpt}")

    cmd_eval = [
        '/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3',
        str(BASE / 'scripts/task84_hgrec_stage3_train.py'),
        '--dataset_name', 'Instruments',
        '--dataset_path', str(BASE / 'HG-Rec/dataset'),
        '--code_path', code_path_arg,
        '--codebook_size', '64', '128', '256', '1',
        '--num_layers', str(args.num_layers),
        '--num_decoder_layers', str(args.num_decoder_layers),
        '--d_model', str(args.d_model),
        '--d_ff', str(args.d_ff),
        '--num_heads', str(args.num_heads),
        '--d_kv', str(args.d_kv),
        '--vocab_size', str(args.vocab_size),
        '--max_len', str(args.max_len),
        '--pad_token_id', '0',
        '--eos_token_id', '0',
        '--device', args.device,
        '--mode', 'test',
        '--save_path', str(out_dir),
        '--log_path', str(log_dir),
        '--seed', str(args.seed),
        '--beam_size', str(args.beam_size),
        '--infer_size', str(args.infer_size),
        '--ckpt_path', str(best_ckpt),
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(BASE / 'HG-Rec') + ':' + env.get('PYTHONPATH', '')
    env['HF_HOME'] = '/home/wlia0047/ar57_scratch/wenyu/hf_models'
    env['TRITON_CACHE_DIR'] = f'/home/wlia0047/.triton/cache_task{args.task_id}_{args.arm_name}_st4'
    os.makedirs(env['TRITON_CACHE_DIR'], exist_ok=True)

    print(f"\nStage 4 test eval (beam={args.beam_size})...")
    t0 = time.time()
    with open(stage4_log, 'w') as f:
        result = subprocess.run(cmd_eval, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(BASE))
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"❌ Stage 4 failed (rc={result.returncode}), check {stage4_log}")
        sys.exit(1)
    print(f"✅ Stage 4 complete in {dt/60:.1f} min")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Task #{args.task_id} {args.arm_name} Stage 3+4 complete")
    print(f"{'='*60}")
    print(f"  Stage 3 log: {stage3_log}")
    print(f"  Stage 4 log: {stage4_log}")
    print(f"  Verdict JSON: verdicts/task{args.task_id}_{args.arm_name}_stage4_beam{args.beam_size}.json")


if __name__ == "__main__":
    main()
