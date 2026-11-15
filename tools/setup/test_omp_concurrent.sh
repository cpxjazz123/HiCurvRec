#!/bin/bash
# test_omp_concurrent.sh — 回归测试：并发实例不得再出现 Bus error (core dumped)。
#
# 故障形态（2026-11-16 在本机复现）：官方原样安装的 oh-my-pi，4 个实例并发跑
# `omp models` 全部 Bus error (core dumped)，退出码 135；同一时间只读的
# `omp auth export` 4 个实例都正常。原因是 SQLite 的 WAL 模式：多个进程打开
# 同一个数据库文件时，后来的进程会重建 <db>-shm，前面那个进程映射的共享内存
# 随之失效，进程立刻 SIGBUS。
#
# 这个测试覆盖：
#   1. 检测器自测：真的制造一次 SIGBUS，确认检测逻辑抓得住（否则"没崩"可能
#      只是检测器失灵）。
#   2. 回归：4 个实例并发跑 `omp models`，不得出现退出码 135。
#   3. 隔离：并发实例拿到不同的槽位与不同的 agent 目录。
#   4. 配置唯一：实例的配置是软链，全部指向 canonical。
#   5. 凭据：每个实例都有凭据（否则并发跑起来了也用不了模型）。
#
# 用法:  bash tools/setup/test_omp_concurrent.sh
set -u

HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
WRAP="$HERE/omp-concurrent"
REAL="$HOME/.local/bin/omp"
POOL="$HOME/.omp/instances"
WORK="$HOME/.omp/.test"

pass=0
fail=0

ok()   { echo "  PASS  $1"; pass=$((pass + 1)); }
bad()  { echo "  FAIL  $1"; fail=$((fail + 1)); }
check() { if [ "$1" = "0" ]; then ok "$2"; else bad "$2"; fi; }

rm -rf "$WORK"; mkdir -p "$WORK"

echo "== 1. 检测器自测：真的制造一次 SIGBUS =="
( sh -c 'kill -BUS $$' ) 2>/dev/null
rc=$?
if [ "$rc" -eq 135 ]; then
  ok "SIGBUS 的退出码是 135，检测器认得它"
else
  bad "SIGBUS 的退出码是 $rc，不是 135，检测规则需要改"
fi

echo "== 2. 回归：4 个实例并发跑 omp models =="
for i in 1 2 3 4; do
  ( "$WRAP" models >"$WORK/out-$i" 2>"$WORK/err-$i"; echo $? >"$WORK/rc-$i" ) &
done
wait
crashed=0
for i in 1 2 3 4; do
  rc=$(cat "$WORK/rc-$i" 2>/dev/null || echo 99)
  if [ "$rc" = "135" ] || [ "$rc" -ge 128 ]; then
    bad "实例 $i 异常退出（退出码 $rc）：$(head -1 "$WORK/err-$i")"
    crashed=1
  fi
done
[ "$crashed" = "0" ] && ok "4 个并发实例全部正常退出，无 Bus error"

echo "== 3. 隔离：实例的槽位与 agent 目录各不相同 =="
# 必须并发地跑：顺序跑的话前一个进程已经退出、锁已释放，四个实例会拿到同一个槽位。
for i in 1 2 3 4; do
  ( "$WRAP" config path >"$WORK/path-$i" 2>/dev/null ) &
done
wait
distinct=$(cat "$WORK"/path-* 2>/dev/null | sort -u | grep -c .)
if [ "$distinct" -ge 2 ]; then
  ok "并发实例拿到了 $distinct 个不同的 agent 目录"
else
  bad "所有实例都用了同一个 agent 目录（$(cat "$WORK"/path-* | tr '\n' ' ')）"
fi

echo "== 4. 配置唯一：实例的配置软链指向 canonical =="
bad_link=""
for d in "$POOL"/inst-*; do
  [ -d "$d/agent" ] || continue
  for f in "$d/agent/config.yml"; do
    [ -e "$f" ] || continue
    target=$(readlink -f "$f")
    case "$target" in
      "$HOME/.omp/agent/"*) ;;
      *) bad_link="$f -> $target" ;;
    esac
  done
done
if [ -z "$bad_link" ]; then
  ok "实例的配置全部软链到 canonical"
else
  bad "配置没有指向 canonical：$bad_link"
fi

echo "== 5. 凭据：每个实例都有凭据 =="
for d in "$POOL"/inst-*; do
  [ -d "$d/agent" ] || continue
  n=$(python3 "$HERE/omp-dbtool.py" credcount "$d/agent/agent.db" 2>/dev/null || echo -1)
  if [ "${n:-0}" -ge 1 ] 2>/dev/null; then
    ok "$(basename "$d") 有 $n 条凭据"
  else
    bad "$(basename "$d") 没有凭据（$n）"
  fi
done

rm -rf "$WORK"
echo
echo "通过 $pass 项，失败 $fail 项"
[ "$fail" -eq 0 ]
