#!/usr/bin/env python3
"""
一次性脚本: 批量替换 logs/task*.log 中所有 task<old_num> 旧编号为新编号.

使用 Task #80 renumber 映射: 104-117 → 23-35, 127-130 → 32-35.

仅替换日志中出现的旧编号引用, 不动文件名 (文件名已在前面 rename 阶段处理).
"""
import re
from pathlib import Path

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
LOGS_DIR = ROOT / 'logs'

MAPPING = {
    '104': '23', '105': '24',
    '107': '25',
    '108': '26', '109': '27', '110': '28', '111': '29', '112': '30',
    '113': '31', '114': '32', '115': '33', '116': '34', '117': '35',
    '127': '32', '128': '33', '129': '34', '130': '35',
}


def replace_refs(text: str) -> str:
    for old, new in MAPPING.items():
        # 先替换 Task #104 / #104 / task104 三种格式
        # 使用负向前瞻/后顾确保不破坏 task1045 等 (但日志里通常不会)
        # ANSI 序列包裹: 先简单替换所有 task104 / Task #104 / #104
        text = text.replace(f'task{old}', f'task{new}')
        text = text.replace(f'Task #{old}', f'Task #{new}')
        # #104 引用 (但要避免匹配 #1045 等)
        text = re.sub(rf'(?<!\d)#{old}(?!\d)', f'#{new}', text)
    return text


def main():
    if not LOGS_DIR.exists():
        print("logs/ 不存在")
        return

    updated = 0
    scanned = 0
    # 只处理 logs/ 顶层 task*.log 文件 (用户编写的日志), 不动 .hydra/ 子树
    for log_file in sorted(LOGS_DIR.glob('task*.log')):
        scanned += 1
        try:
            content = log_file.read_text(encoding='utf-8')
        except (UnicodeDecodeError, PermissionError):
            continue
        new_content = replace_refs(content)
        if new_content != content:
            log_file.write_text(new_content, encoding='utf-8')
            print(f"  UPDATED: {log_file.name}")
            updated += 1

    print(f"\n扫描 {scanned} 个日志文件, 更新 {updated} 个")


if __name__ == '__main__':
    main()