#!/bin/sh
# Regression test for omp-local: where oh-my-pi's state lives, and how many
# instances may run on one profile at the same time.
#
# The state lives in oh-my-pi's OWN home directory, $HOME/.omp, which is on NFS
# and therefore visible from every node of the cluster.  There is no
# node-local installation step.
#
#   $HOME/.omp/agent/                       the default profile's agent dir
#   $HOME/.omp/profiles/<profile>/agent/    every other profile's agent dir
#   $HOME/.omp/omp-slots/<profile>/<slot>/  private agent dir of a 2nd+
#                                           instance of that profile
#   $HOME/.omp/omp-slots/<profile>/.locks/  ownership locks of the slots
#
# Two invariants are guarded:
#
#   * No two live processes may open the same SQLite file.  The databases live
#     in the agent dir, so each live instance needs one of its own.
#   * The CONFIGURATION must not be lost: the canonical agent dir of the
#     default profile is $HOME/.omp/agent, and its config.yml carries the
#     model roles.  A wrapper that keeps its own copy of the profile tree
#     silently starts omp without any modelRoles (cases 1 and 5).
#
# Every case below must pass; the script exits non-zero if any fails.
#
# Usage:  tools/setup/test_omp_local.sh [profile-name]
#
# Optional: set OMP_TEST_PEER=<host> to also run the cross-node case (case 11).

set -u

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_WRAPPER=$HERE/omp-local
WRAPPER=${OMP_LOCAL_WRAPPER:-$REPO_WRAPPER}
PROF=${1:-selftest_conc}
OMP_HOME=$HOME/.omp
ROOT=$OMP_HOME/profiles/$PROF
AGENT=$ROOT/agent
SLOTS=$OMP_HOME/omp-slots/$PROF
SLOTLOCKS=$SLOTS/.locks
LOCK=$SLOTS/.role.lock
TMP=$HERE/.omp_local_test_tmp
N=4
PEER=${OMP_TEST_PEER:-}

pass=0
fail=0
skip=0

cleanup() {
  # Only the test's own profile is removed.  $HOME/.omp/agent (the default
  # profile) is read-only for this test and must survive it.
  /bin/rm -rf "$ROOT" "$SLOTS" "$TMP"
}
trap cleanup EXIT INT TERM

ok()  { pass=$((pass + 1)); echo "  PASS  $1"; }
bad() { fail=$((fail + 1)); echo "  FAIL  $1"; }
skipt() { skip=$((skip + 1)); echo "  SKIP  $1"; }

MYHOST=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)

# Hold the profile lock in the background for <seconds>.  The holder `exec`s the
# sleep, so $HOLDER_PID IS the process holding the lock and killing it releases
# the lock immediately.  Must NOT be called inside a command substitution: the
# background job would inherit the substitution pipe and the caller would block
# until the lock expired.
#
# The lock file is opened read-write WITHOUT O_TRUNC/O_APPEND: over NFS a
# contender that opens it that way does not conflict with the lock the holder
# already has, which would make two instances believe they own the profile.
HOLDER_PID=""
hold_lock() {
  [ -e "$LOCK" ] || touch "$LOCK"
  ( flock 9; exec sleep "$1" ) 9<>"$LOCK" &
  HOLDER_PID=$!
}

release_lock() {
  if [ -n "$HOLDER_PID" ]; then
    kill "$HOLDER_PID" 2>/dev/null
    wait "$HOLDER_PID" 2>/dev/null
    HOLDER_PID=""
  fi
}

# Hold the lock of a slot for <seconds>, as a live owner would.
hold_slot_lock() {  # hold_slot_lock <slotname> <seconds>
  [ -e "$SLOTLOCKS/$1.lock" ] || touch "$SLOTLOCKS/$1.lock"
  ( flock 9; exec sleep "$2" ) 9<>"$SLOTLOCKS/$1.lock" &
  HOLDER_PID=$!
}

# Is the lock of <file> free?  Returns 0 when it could be taken, 1 when it is
# held.  The file must be opened read-write, otherwise the check reports EBADF
# instead of EWOULDBLOCK and every lock would look free.
lock_is_free() {
  ( exec 9<>"$1"; flock -n 9 ) 2>/dev/null
}

# The agent dir the wrapper picked, as reported by `omp config path`.
agent_dir_of() {
  "$WRAPPER" "$1" config path 2>/dev/null | grep -E '^/' | tail -1
}

mkdir -p "$TMP"

echo "== 用例 0: wrapper 副本一致，且不含节点本地路径 =="
h_repo=$(sha256sum "$REPO_WRAPPER" | cut -d' ' -f1)
for copy in "$HOME/bin/omp-local"; do
  if [ ! -f "$copy" ]; then
    bad "缺少副本 $copy"
  elif [ "$(sha256sum "$copy" | cut -d' ' -f1)" = "$h_repo" ]; then
    ok "$copy 与仓库副本一致"
  else
    bad "$copy 与仓库副本不一致（源码漂移）"
  fi
done
if grep -n '/var/tmp' "$REPO_WRAPPER" >/dev/null 2>&1; then
  bad "wrapper 里仍有 /var/tmp 字面量：$(grep -n '/var/tmp' "$REPO_WRAPPER" | head -3)"
else
  ok "wrapper 不含 /var/tmp 字面量"
fi
# 锁文件必须以 O_RDWR 打开：NFS 上以 O_TRUNC/O_APPEND 打开的竞争方，
# 其锁请求不会与持有方的锁冲突，于是两个节点会同时自认主角色。
if grep -nE 'exec [89]>{1,2}"' "$REPO_WRAPPER" >/dev/null 2>&1; then
  bad "锁文件被以截断/追加方式打开（跨节点互斥会失效）: $(grep -nE 'exec [89]>{1,2}\"' "$REPO_WRAPPER" | head -3)"
else
  ok "锁文件以 O_RDWR 打开，不含 O_TRUNC/O_APPEND"
fi
# wrapper 必须留在 omp 自己的家目录里，不能另造一套 profile 树：另造的那套
# 不会带上 $HOME/.omp/agent/config.yml 里的 modelRoles，omp 启动就没有模型。
if grep -nE 'PI_CONFIG_DIR=' "$REPO_WRAPPER" | grep -v 'PI_CONFIG_DIR=\.omp' >/dev/null 2>&1; then
  bad "wrapper 把 PI_CONFIG_DIR 指到了 omp 之外: $(grep -nE 'PI_CONFIG_DIR=' "$REPO_WRAPPER" | head -3)"
else
  ok "wrapper 使用 omp 自己的家目录（PI_CONFIG_DIR=.omp）"
fi

echo "== 用例 1: default profile 用 omp 自己的 agent 目录，且模型配置还在 =="
if [ -d "$OMP_HOME/agent" ]; then
  ok "default profile 的 agent 目录是 $OMP_HOME/agent"
else
  bad "缺少 $OMP_HOME/agent"
fi
if [ -f "$OMP_HOME/agent/config.yml" ] && grep -q 'modelRoles' "$OMP_HOME/agent/config.yml"; then
  ok "default profile 的 config.yml 仍在（modelRoles 未丢）"
else
  bad "default profile 的 config.yml 丢了 modelRoles: $(cat "$OMP_HOME/agent/config.yml" 2>&1 | head -3)"
fi
d=$(agent_dir_of default)
case "$d" in
  "$OMP_HOME/agent") ok "default 实例使用规范 agent 目录" ;;
  "$OMP_HOME"/omp-slots/default/*) ok "default 已被占用，实例落入私有 slot: $d" ;;
  *) bad "default 实例的 agent 目录不对: $d" ;;
esac

echo "== 用例 2: 其它 profile 用 omp 自己的 profile 目录 =="
mkdir -p "$AGENT" "$SLOTS" "$SLOTLOCKS"
d=$(agent_dir_of "$PROF")
if [ "$d" = "$AGENT" ]; then
  ok "profile $PROF 使用 $d"
else
  bad "profile $PROF 的目录不对: $d（期望 $AGENT）"
fi
# Marker config, written before any slot exists so that every slot created
# below inherits it through the symlink.
printf 'setupVersion: 3\nmodelRoles:\n  default: selftest-marker-model\n' >"$AGENT/config.yml"

echo "== 用例 3: 规范实例存活时，第二个实例不再被拒绝 =="
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
  "$SLOTS"/*) ok "第二个实例落入私有 slot: $out" ;;
  *) bad "第二个实例没有落入 slot，而是用了 $out" ;;
esac
case "$err" in
  *"private slot"*) ok "stderr 说明了 slot 归属" ;;
  *) bad "stderr 没有 slot 提示: $err" ;;
esac

echo "== 用例 4: $N 个真正重叠的并发实例全部成功，且 agent 目录互不相同 =="
# `omp bash -c ... sleep` keeps each instance alive for the whole window, so the
# $N instances really do overlap -- this is the invariant that matters: no two
# live instances may share one agent directory (and therefore one SQLite file).
hold_lock 45
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
if grep -q "^$SLOTS/" "$TMP/dirs" && ! grep -qx "$AGENT" "$TMP/dirs"; then
  ok "所有并发实例都落在 slot 里，没有碰规范目录"
else
  bad "并发实例落到了规范目录，槽位隔离失效"
fi
if grep -qv "^$OMP_HOME/" "$TMP/dirs"; then
  bad "有 agent 目录不在 \$HOME 下: $(grep -v "^$OMP_HOME/" "$TMP/dirs" | head -3)"
else
  ok "所有 agent 目录都在 \$HOME（$OMP_HOME）之下"
fi

echo "== 用例 5: 槽位与规范目录共享配置（软链，不是复制）=="
slot=$(head -1 "$TMP/dirs")
if [ -L "$slot/config.yml" ] && [ "$(readlink "$slot/config.yml")" = "$AGENT/config.yml" ]; then
  ok "槽位 config.yml 软链到规范目录"
else
  bad "槽位 config.yml 不是指向规范目录的软链: $(ls -l "$slot/config.yml" 2>&1)"
fi
hold_lock 8
sleep 1
listing=$("$WRAPPER" "$PROF" config list --scope=global 2>/dev/null)
release_lock
case "$listing" in
  *selftest-marker-model*) ok "槽位实例读到共享配置" ;;
  *) bad "槽位实例没有读到共享配置" ;;
esac

echo "== 用例 6: SQLite 文件不被两个实例共用 =="
if [ -f "$AGENT/agent.db" ] && [ -f "$slot/agent.db" ]; then
  i1=$(ls -i "$AGENT/agent.db" | cut -d' ' -f1)
  i2=$(ls -i "$slot/agent.db" | cut -d' ' -f1)
  if [ "$i1" != "$i2" ]; then
    ok "规范目录与槽位的 agent.db 是不同 inode（$i1 vs $i2）"
  else
    bad "两个实例共用同一个 agent.db"
  fi
else
  bad "缺少 agent.db，无法校验隔离"
fi

echo "== 用例 7: 存活实例持有的槽位不会被抢占 =="
mkdir -p "$SLOTS/live-owner" "$SLOTLOCKS"
hold_slot_lock live-owner 8
sleep 1
hold_lock 8
sleep 1
d=$(agent_dir_of "$PROF")
release_lock
case "$d" in
  *live-owner*) bad "抢占了存活实例持有的槽位: $d" ;;
  *) ok "存活实例的槽位未被抢占（选到了 $d）" ;;
esac

echo "== 用例 8: 空闲（陈旧）的槽位被复用，而不是无限堆积 =="
# 先把其它槽位清掉，只留 live-owner 和本用例新建的空闲槽位，
# 这样 wrapper 只有一个可选目标，断言才是确定的。
for d in "$SLOTS"/*; do
  case "${d##*/}" in .*|live-owner|free-owner) continue ;; esac
  /bin/rm -rf "$d"
done
for l in "$SLOTLOCKS"/*; do
  case "${l##*/}" in live-owner.lock) continue ;; esac
  /bin/rm -f "$l"
done
mkdir -p "$SLOTS/free-owner"
before=$(/bin/ls -1 "$SLOTS" 2>/dev/null | wc -l)
hold_lock 8
sleep 1
d=$(agent_dir_of "$PROF")
release_lock
case "$d" in
  *free-owner*) ok "空闲槽位被复用: $d" ;;
  *) bad "空闲槽位没有被复用（选到了 $d）" ;;
esac
after=$(/bin/ls -1 "$SLOTS" 2>/dev/null | wc -l)
if [ "$after" -le "$before" ]; then
  ok "槽位数量未增长（$before -> $after），没有新造槽位"
else
  bad "槽位数量增长（$before -> $after），没有复用空闲槽位"
fi

echo "== 用例 9: 槽位的归属锁不放在 omp 的 agent 目录里 =="
# omp 自己拥有 agent 目录，会清理它不认识的文件；锁文件必须放在别处，
# 否则锁文件被删后重新创建会换 inode，新来的实例就拿不到同一把锁。
if [ -d "$SLOTLOCKS" ]; then
  ok "归属锁目录在 $SLOTLOCKS（agent 目录之外）"
else
  bad "缺少 $SLOTLOCKS 锁目录"
fi
if find "$SLOTS" -mindepth 2 -maxdepth 2 -name '*.lock' -not -path "$SLOTLOCKS/*" 2>/dev/null | grep -q .; then
  bad "有 .lock 文件落在 agent 目录内: $(find "$SLOTS" -mindepth 2 -maxdepth 2 -name '*.lock' -not -path "$SLOTLOCKS/*" | head -3)"
else
  ok "agent 目录内没有 .lock 文件"
fi

echo "== 用例 10: 活着的实例确实持有自己的槽位锁 =="
# 挡住规范目录，逼这个实例进 slot，然后验证它真的持有那把槽位锁
# （锁必须跟着进程活过 exec，否则别的实例就能抢走同一个槽位）。
hold_lock 30
( "$WRAPPER" "$PROF" bash -c 'echo "AGENTDIR=$PI_CODING_AGENT_DIR"; sleep 10' \
    >"$TMP/out.lockcheck" 2>/dev/null ) &
CHECK_PID=$!
sleep 5
lockdir=$(sed -n 's/^AGENTDIR=//p' "$TMP/out.lockcheck" 2>/dev/null | head -1)
lockname=${lockdir##*/}
if [ -z "$lockname" ]; then
  bad "实例没有报告自己的 agent 目录"
elif [ "$lockname" = "agent" ]; then
  bad "实例没有落进 slot（拿到了规范目录），用例无效"
else
  if lock_is_free "$SLOTLOCKS/$lockname.lock"; then
    bad "槽位锁没有被实例持有（$lockname 可被抢占）"
  else
    ok "活着的实例持有槽位锁 $lockname.lock"
  fi
fi
wait "$CHECK_PID" 2>/dev/null
release_lock

echo "== 用例 11: 跨节点并发（可选，需要 OMP_TEST_PEER）=="
if [ -z "$PEER" ]; then
  skipt "未设置 OMP_TEST_PEER，跳过跨节点用例"
else
  ( "$WRAPPER" "$PROF" bash -c 'echo "AGENTDIR=$PI_CODING_AGENT_DIR"; sleep 12' \
      >"$TMP/out.peer" 2>/dev/null ) &
  PEER_PID=$!
  sleep 5
  peer_out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    "'$WRAPPER' '$PROF' bash -c 'echo PEERDIR=\$PI_CODING_AGENT_DIR; sleep 2'" 2>&1 |
    grep -E 'PEERDIR=' | sed 's/^PEERDIR=//' | head -1)
  wait "$PEER_PID" 2>/dev/null
  local_dir=$(sed -n 's/^AGENTDIR=//p' "$TMP/out.peer" 2>/dev/null | head -1)
  if [ -z "$peer_out" ]; then
    bad "对端节点没有报出 agent 目录"
  elif [ "$peer_out" = "$local_dir" ]; then
    bad "对端节点与本节点共用了同一个 agent 目录: $peer_out"
  else
    ok "对端节点（$PEER）拿到了不同的 agent 目录: $peer_out"
  fi
fi

echo
echo "通过 $pass 项，失败 $fail 项，跳过 $skip 项"
[ "$fail" -eq 0 ] || exit 1
