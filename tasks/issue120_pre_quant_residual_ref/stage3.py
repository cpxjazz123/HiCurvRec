#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #120 — Stage3 stub.

R34 严格: 4 脚本必须存在. Issue #120 仅 Stage2 几何诊断, 不跑 Stage3 T5.
R36 严格: 不修改 Stage3 forward path (issue #120 显式要求).
"""
import os
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
print(f"[issue120-stage3] Issue #120 不跑 Stage3 T5 (issue #120 明确: 仅 Stage2 几何诊断)")
print(f"[issue120-stage3] 跳过 Stage3 — 不修改 Stage3 forward path")
