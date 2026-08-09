"""Issue #97 Stage 4: ActionPiece (Hou et al., ICML 2025) 复现 — 上下文感知 action tokenization。

官方代码: github.com/google-deepmind/action_piece
预检查官方代码是否已下载; 缺失则提示 git clone。
"""
import os
import sys

CLONE_TARGET = "/fs04/ar57/wenyu/action_piece"


def main():
    print("[Stage4 ActionPiece] precheck")
    if os.path.isdir(CLONE_TARGET):
        print(f"  repo: {CLONE_TARGET} OK")
    else:
        print(f"  repo: {CLONE_TARGET} MISSING -> 需要 git clone https://github.com/google-deepmind/action_piece")
        sys.exit(1)


if __name__ == "__main__":
    main()
