"""Issue #97 Stage 2: LETTER (Wang et al., CIKM 2024) 复现评估 (t5-base, recall@10 口径)。

LETTER = RQ-VAE + 对比对齐 + diversity loss 的 learnable tokenizer,
instantiation 在 LETTER-TIGER (LETTER tokenizer + T5 解码器)。
已有 t5-base 200ep ckpt (checkpoint-49278, 历史 hit@10=0.0953),
本 stage 用 recall@10 口径复跑评估验证。
"""
import os
import subprocess
import sys

REPO = "/fs04/ar57/wenyu/LETTER/LETTER-TIGER"
CKPT = f"{REPO}/ckpt/Instruments_t5base_letter_fixed"
DATA = "/fs04/ar57/wenyu/LETTER/data"
RESULTS = "/fs04/ar57/wenyu/GeneRec/tasks/4baseline_repro/results"


def main():
    os.makedirs(RESULTS, exist_ok=True)
    assert os.path.isdir(CKPT), f"LETTER ckpt missing: {CKPT}"
    print(f"[Stage2 LETTER] eval ckpt={CKPT}")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        sys.executable, "test.py",
        "--gpu_id", "0",
        "--ckpt_path", CKPT,
        "--dataset", "Instruments",
        "--data_path", DATA,
        "--results_file", f"{RESULTS}/letter_t5base_best49278_recall.json",
        "--test_batch_size", "32",
        "--num_beams", "20",
        "--test_prompt_ids", "0",
        "--index_file", ".index.json",
        "--metrics", "recall@5,recall@10,recall@20,ndcg@5,ndcg@10,ndcg@20",
    ]
    print("CMD:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO, env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
