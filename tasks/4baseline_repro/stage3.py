"""Issue #97 Stage 3: LC-Rec (Zheng et al., 2024) 复现 — LLaMA LoRA 微调 + 评估。

LC-Rec 官方实现位于 LETTER-LC-Rec, 需要 LLaMA base model。
预检查 LLaMA 权重可用性 + user.json 数据, 缺失则报告。
"""
import os
import sys

REPO = "/fs04/ar57/wenyu/LETTER/LETTER-LC-Rec"
DATA = "/fs04/ar57/wenyu/LETTER/data"


def main():
    print("[Stage3 LC-Rec] precheck")
    ok = True
    if not os.path.isdir(REPO):
        print("  FAIL: LC-Rec repo missing"); ok = False
    else:
        print(f"  repo: {REPO} OK")
    # LLaMA base model 检查
    llama = os.path.expanduser("/fs04/ar57/wenyu/LETTER/ckpt/LLaMA-7b")
    if os.path.isdir(llama):
        print(f"  LLaMA-7b: {llama} OK")
    else:
        print(f"  LLaMA-7b: {llama} MISSING")
    # user.json 数据检查
    for d in ["Instruments"]:
        p = f"{DATA}/{d}/{d}.user.json"
        if os.path.isfile(p):
            print(f"  user.json: {p} OK")
        else:
            print(f"  user.json: {p} MISSING (LC-Rec 需要)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
