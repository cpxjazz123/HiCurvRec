#!/bin/sh
# Regression test for omp-local: how many concurrent omp instances may run on
# one profile.  Before 2026-11-15 the wrapper took a per-profile flock and
# refused the second start with exit 3 ("profile 'x' is already in use"), so a
# second `omp` on the same profile simply could not start.  The lock is now a
# role selector: the first instance owns the canonical agent directory, every
# further instance transparently lands in a private slot with its own SQLite
# files, and the shared configuration is symlinked into that slot.
#
# Every case below must pass; the script exits non-zero if any fails.
#
# Usage:  tools/setup/test_omp_local.sh [profile-name]

set -u

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_WRAPPER=$HERE/omp-local
WRAPPER=${OMP_LOCAL_WRAPPER:-$REPO_WRAPPER}
LOCAL=/var/tmp/omp_local
PROF=${1:-selftest_conc}
ROOT=$LOCAL/roots/$PROF
HOME_LINK=$HOME/omp_local_$PROF
TMP=$HERE/.omp_local_test_tmp
N=8

pass=0
fail=0

cleanup() {
  rm -rf "$ROOT" "$HOME_LINK" "$LOCAL/locks/$PROF.lock" "$TMP"
}
trap cleanup EXIT INT TERM

ok()  { pass=$((pass + 1)); echo "  PASS  $1"; }
bad() { fail=$((fail + 1)); echo "  FAIL  $1"; }

# Hold the profile lock in the background for <seconds>.  The holder `exec`s the
# sleep, so $HOLDER_PID IS the process holding the lock and killing it releases
# the lock immediately.  Must NOT be called inside a command substitution: the
# background job would inherit the substitution pipe and the caller would block
# until the lock expired.
HOLDER_PID=""
hold_lock() {
  ( flock 9; exec sleep "$1" ) 9>"$LOCAL/locks/$PROF.lock" &
  HOLDER_PID=$!
}

release_lock() {
  if [ -n "$HOLDER_PID" ]; then
    kill "$HOLDER_PID" 2>/dev/null
    wait "$HOLDER_PID" 2>/dev/null
    HOLDER_PID=""
  fi
}

# The agent dir the wrapper picked, as reported by `omp config path`.
agent_dir_of() {
  "$WRAPPER" "$1" config path 2>/dev/null | grep -E '^/' | tail -1
}

mkdir -p "$TMP"

echo "== 用例 0: 三份 wrapper 副本内容一致 =="
h_repo=$(sha256sum "$REPO_WRAPPER" | cut -d' ' -f1)
for copy in "$HOME/bin/omp-local" "$LOCAL/bin/omp-local"; do
  if [ ! -f "$copy" ]; then
    bad "缺少副本 $copy"
  elif [ "$(sha256sum "$copy" | cut -d' ' -f1)" = "$h_repo" ]; then
    ok "$copy 与仓库副本一致"
  else
    bad "$copy 与仓库副本不一致（源码漂移）"
  fi
done

echo "== 用例 1: 没有竞争时，第一个实例使用规范 agent 目录 =="
d=$(agent_dir_of "$PROF")
if [ "$d" = "$ROOT/agent" ]; then
  ok "primary 使用 $d"
else
  bad "primary 目录不对: $d（期望 $ROOT/agent）"
fi
# Marker config, written before any slot exists so that every slot created
# below inherits it through the symlink.
printf 'setupVersion: 3\nmodelRoles:\n  default: selftest-marker-model\n' >"$ROOT/agent/config.yml"

echo "== 用例 2: 规范实例存活时，第二个实例不再被拒绝 =="
hold_lock 6
sleep 1
out=$("$WRAPPER" "$PROF" config path 2>"$TMP/err2")
rc=$?
err=$(cat "$TMP/err2")
release_lock
if [ "$rc" -eq 0 ]; then
  ok "第二个实例启动成功（退出码 0，旧行为是 3）"
else
  bad "第二个实例退出码 $rc（旧行为 3 = 被拒绝）"
fi
case "$out" in
  "$ROOT"/agents/*) ok "第二个实例落入私有 slot: $out" ;;
  *) bad "第二个实例没有落入 slot，而是用了 $out" ;;
esac
case "$err" in
  *"private slot"*) ok "stderr 说明了 slot 归属" ;;
  *) bad "stderr 没有 slot 提示: $err" ;;
esac

echo "== 用例 3: $N 个真正重叠的并发实例全部成功，且 agent 目录互不相同 =="
# `omp bash -c ... sleep` keeps each instance alive for the whole window, so the
# $N instances really do overlap -- this is the invariant that matters: no two
# live instances may share one agent directory (and therefore one SQLite file).
hold_lock 90
sleep 1
pids=""
i=1
while [ "$i" -le "$N" ]; do
  ( "$WRAPPER" "$PROF" bash -c 'echo "AGENTDIR=$PI_CODING_AGENT_DIR"; sleep 6' \
      >"$TMP/out.$i" 2>"$TMP/err.$i"; echo $? >"$TMP/rc.$i" ) &
  pids="$pids $!"
  i=$((i + 1))
done
for p in $pids; do wait "$p"; done
release_lock

all_ok=1
i=1
while [ "$i" -le "$N" ]; do
  rc=$(cat "$TMP/rc.$i" 2>/dev/null || echo 99)
  if [ "$rc" != "0" ]; then
    bad "第 $i 个实例退出码 $rc"
    all_ok=0
  fi
  i=$((i + 1))
done
[ "$all_ok" = "1" ] && ok "$N 个重叠实例全部启动成功"

grep -h '^AGENTDIR=' "$TMP"/out.* 2>/dev/null | sed 's/^AGENTDIR=//' | sort >"$TMP/dirs"
total=$(wc -l <"$TMP/dirs")
uniq=$(sort -u "$TMP/dirs" | wc -l)
if [ "$total" -eq "$N" ] && [ "$uniq" -eq "$N" ]; then
  ok "$N 个重叠实例拿到 $uniq 个互不相同的 agent 目录"
else
  bad "agent 目录重复: 共 $total 个，去重后 $uniq 个"
  sort "$TMP/dirs" | uniq -c | sort -rn | head -3
fi
if grep -q "^$ROOT/agents/" "$TMP/dirs" && ! grep -qx "$ROOT/agent" "$TMP/dirs"; then
  ok "所有并发实例都落在 slot 里，没有碰规范目录"
else
  bad "并发实例落到了规范目录，槽位隔离失效"
fi

echo "== 用例 4: 槽位里的配置与规范目录共享 =="
slot=$(head -1 "$TMP/dirs")
if [ -L "$slot/config.yml" ] && [ "$(readlink "$slot/config.yml")" = "$ROOT/agent/config.yml" ]; then
  ok "槽位 config.yml 软链到规范目录"
else
  bad "槽位 config.yml 不是指向规范目录的软链"
fi
hold_lock 8
sleep 1
listing=$("$WRAPPER" "$PROF" config list --scope=global 2>/dev/null)
release_lock
case "$listing" in
  *selftest-marker-model*) ok "槽位实例读到共享配置" ;;
  *) bad "槽位实例没有读到共享配置" ;;
esac

echo "== 用例 5: SQLite 文件不被两个实例共用 =="
if [ -f "$ROOT/agent/agent.db" ] && [ -f "$slot/agent.db" ]; then
  i1=$(ls -i "$ROOT/agent/agent.db" | cut -d' ' -f1)
  i2=$(ls -i "$slot/agent.db" | cut -d' ' -f1)
  if [ "$i1" != "$i2" ]; then
    ok "规范目录与槽位的 agent.db 是不同 inode（$i1 vs $i2）"
  else
    bad "两个实例共用同一个 agent.db"
  fi
else
  bad "缺少 agent.db，无法校验隔离"
fi

echo "== 用例 6: 存活进程持有的槽位不会被抢占 =="
mkdir -p "$ROOT/agents/live-owner" "$ROOT/agents/.owners"
printf '%s\n' "$$" >"$ROOT/agents/.owners/live-owner"   # 测试脚本自身：存活中
hold_lock 8
sleep 1
d=$(agent_dir_of "$PROF")
release_lock
case "$d" in
  *live-owner*) bad "抢占了存活进程持有的槽位: $d" ;;
  *) ok "存活进程的槽位未被抢占（选到了 $d）" ;;
esac

echo "== 用例 7: 陈旧槽位被复用，而不是无限堆积 =="
before=$(ls -1 "$ROOT/agents" 2>/dev/null | wc -l)
hold_lock 8
sleep 1
d=$(agent_dir_of "$PROF")
release_lock
after=$(ls -1 "$ROOT/agents" 2>/dev/null | wc -l)
if [ "$after" -le "$before" ]; then
  ok "槽位数量未增长（$before -> $after），陈旧槽位被复用"
else
  bad "槽位数量增长（$before -> $after），没有复用陈旧槽位"
fi

echo
echo "通过 $pass 项，失败 $fail 项"
[ "$fail" -eq 0 ] || exit 1
