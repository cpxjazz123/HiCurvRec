#!/bin/sh
# Regression test for omp-local: one configuration, any number of instances.
#
# The state lives in oh-my-pi's OWN home directory, $HOME/.omp, which is on NFS
# and therefore visible from every node of the cluster.  There is no
# node-local installation step.
#
#   $HOME/.omp/agent/                     the configuration's single source of
#                                         truth, and the agent dir of the ONE
#                                         canonical instance
#   $HOME/.omp/omp-slots/<ns>/<slot>/     private agent dir of every further
#                                         instance
#   $HOME/.omp/omp-slots/<ns>/.locks/     ownership locks of the slots
#   $HOME/.omp/omp-slots/.canonical.lock  ownership lock of the canonical dir
#
# Three invariants are guarded:
#
#   * No two live processes may open the same SQLite file.  The databases live
#     in the agent dir, so each live instance needs one of its own.
#   * EVERY INSTANCE READS THE SAME CONFIGURATION.  A profile used to be a
#     second tree of its own ($HOME/.omp/profiles/<p>/agent) with its own copy
#     of the configuration; the copies drifted apart and a stale database from
#     an older omp version in one of them crashed with SIGBUS (sqlite rewrites
#     the main database file while it migrates the schema, which on NFS
#     invalidates the mapping of every process holding the file open).
#   * The CONFIGURATION must not be lost: it lives in $HOME/.omp/agent, and
#     its config.yml carries the model roles.
#
# Every case below must pass; the script exits non-zero if any fails.
#
# Usage:  tools/setup/test_omp_local.sh [namespace]
#
# Optional: set OMP_TEST_PEER=<host> to also run the cross-node case.

set -u

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_WRAPPER=$HERE/omp-local
WRAPPER=${OMP_LOCAL_WRAPPER:-$REPO_WRAPPER}
NS=${1:-selftest_conc}
OMP_HOME=$HOME/.omp
CANONICAL=$OMP_HOME/agent
SLOTS=$OMP_HOME/omp-slots
NSSLOTS=$SLOTS/$NS
SLOTLOCKS=$NSSLOTS/.locks
LOCK=$SLOTS/.canonical.lock
TMP=$HERE/.omp_local_test_tmp
N=4
PEER=${OMP_TEST_PEER:-}

pass=0
fail=0
skip=0

cleanup() {
  # Only the test's own namespace is removed.  $HOME/.omp/agent (the
  # configuration and the canonical instance) is read-only for this test and
  # must survive it.
  /bin/rm -rf "$NSSLOTS" "$TMP"
}
trap cleanup EXIT INT TERM

ok()  { pass=$((pass + 1)); echo "  PASS  $1"; }
bad() { fail=$((fail + 1)); echo "  FAIL  $1"; }
skipt() { skip=$((skip + 1)); echo "  SKIP  $1"; }

MYHOST=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)

# Hold the canonical lock in the background for <seconds>.  The holder `exec`s
# the sleep, so $HOLDER_PID IS the process holding the lock and killing it
# releases the lock immediately.  Must NOT be called inside a command
# substitution: the background job would inherit the substitution pipe and the
# caller would block until the lock expired.
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

# Number of rows in a table of a database, read through a COPY of the file so
# that the shared-memory file of a live instance is never touched.
rows_in() {  # rows_in <dbfile> <table>
  [ -f "$1" ] || { echo "-1"; return 0; }
  cp "$1" "$TMP/probe.db" 2>/dev/null || { echo "-1"; return 0; }
  python3 - "$TMP/probe.db" "$2" <<'PY' 2>/dev/null || { echo "-1"; exit 0; }
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
try:
    print(conn.execute("SELECT count(*) FROM %s" % sys.argv[2]).fetchone()[0])
except Exception:
    print("-1")
finally:
    conn.close()
PY
}

mkdir -p "$TMP"

echo "== 用例 1: wrapper 副本一致，且不含节点本地路径 =="
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
if [ -f "$HERE/omp-entry" ]; then
  h_entry=$(sha256sum "$HERE/omp-entry" | cut -d' ' -f1)
  if [ ! -f "$HOME/bin/omp" ]; then
    bad "缺少 $HOME/bin/omp（PATH 里的 omp 必须是 wrapper，否则裸 omp 会与别的实例共用 agent 目录）"
  elif [ "$(sha256sum "$HOME/bin/omp" | cut -d' ' -f1)" = "$h_entry" ]; then
    ok "$HOME/bin/omp 与仓库副本一致"
  else
    bad "$HOME/bin/omp 与仓库副本不一致（源码漂移）"
  fi
fi
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

echo "== 用例 2: PATH 里的 omp 是 wrapper，真二进制不在 PATH 上 =="
# SQLite 在 WAL 模式下把 "<db>-shm" mmap 进进程；第二个进程打开同一个库时会
# 重建那个文件，先前那个进程随即收到 SIGBUS（"Bus error at address ..."）。
# 因此两个实例绝不能共用一个 agent 目录 —— 裸 omp 不受 slot 隔离保护，必须
# 从 PATH 上就把它挡掉。
for entry in "$HOME/bin/omp" "$HOME/.local/bin/omp"; do
  if [ ! -f "$entry" ]; then
    bad "缺少 $entry（裸 omp 会与别的实例共用同一个 agent 目录）"
  elif [ "$(head -c 2 "$entry" 2>/dev/null)" = '#!' ]; then
    ok "$entry 是 shell wrapper"
  else
    bad "$entry 不是 wrapper：裸 omp 会与别的实例共用同一个 agent 目录"
  fi
done
if [ -x "$HOME/.local/bin/omp.bin" ]; then
  bad "PATH 上仍有裸二进制 $HOME/.local/bin/omp.bin"
else
  ok "PATH 上没有裸二进制"
fi
if [ -f "$HOME/.omp/bin/omp.real" ] && [ "$(head -c 2 "$HOME/.omp/bin/omp.real" 2>/dev/null)" != '#!' ]; then
  ok "真二进制在 $HOME/.omp/bin/omp.real"
else
  bad "缺少真二进制 $HOME/.omp/bin/omp.real"
fi
mkdir -p "$CANONICAL"
d=$("$HOME/bin/omp" --profile "$NS" config path 2>/dev/null | grep -E '^/' | tail -1)
case "$d" in
  "$CANONICAL") ok "--profile 被映射到规范目录: $d" ;;
  "$NSSLOTS"/*) ok "--profile 被映射到私有 slot: $d" ;;
  *) bad "--profile 处理不对: $d" ;;
esac

echo "== 用例 3: 配置只有一份真源，且模型配置还在 =="
if [ -d "$CANONICAL" ]; then
  ok "规范目录是 $CANONICAL"
else
  bad "缺少 $CANONICAL"
fi
if [ -f "$CANONICAL/config.yml" ] && grep -q 'modelRoles' "$CANONICAL/config.yml"; then
  ok "config.yml 仍在（modelRoles 未丢）"
else
  bad "config.yml 丢了 modelRoles: $(cat "$CANONICAL/config.yml" 2>&1 | head -3)"
fi
d=$(agent_dir_of default)
case "$d" in
  "$CANONICAL") ok "default 实例使用规范目录" ;;
  "$OMP_HOME"/omp-slots/default/*) ok "default 已被占用，实例落入私有 slot: $d" ;;
  *) bad "default 实例的 agent 目录不对: $d" ;;
esac

echo "== 用例 4: 不再有 profile 树，所有实例都用同一份配置 =="
# 曾经每个 profile 有自己的 $HOME/.omp/profiles/<p>/agent，于是配置与数据库
# 各有一份并逐渐分叉（旧数据库迁移时重写主文件 → NFS 上 SIGBUS）。
if grep -n '^[^#]*profiles/' "$REPO_WRAPPER" >/dev/null 2>&1; then
  bad "wrapper 仍在使用 profile 树: $(grep -n '^[^#]*profiles/' "$REPO_WRAPPER" | head -3)"
else
  ok "wrapper 不再引用 profile 树"
fi
d=$(agent_dir_of "$NS")
case "$d" in
  "$CANONICAL"|"$NSSLOTS"/*) ok "实例 $NS 使用 $d" ;;
  *) bad "实例 $NS 的目录不对: $d" ;;
esac

echo "== 用例 5: 规范实例存活时，第二个实例不再被拒绝 =="
hold_lock 6
sleep 1
out=$("$WRAPPER" "$NS" config path 2>"$TMP/err2")
rc=$?
err=$(cat "$TMP/err2")
release_lock
if [ "$rc" -eq 0 ]; then
  ok "第二个实例启动成功（退出码 0，旧行为是 3）"
else
  bad "第二个实例退出码 $rc（旧行为 3 = 被拒绝）"
fi
case "$out" in
  "$NSSLOTS"/*) ok "第二个实例落入私有 slot: $out" ;;
  *) bad "第二个实例没有落入 slot，而是用了 $out" ;;
esac
case "$err" in
  *"private slot"*) ok "stderr 说明了 slot 归属" ;;
  *) bad "stderr 没有 slot 提示: $err" ;;
esac

echo "== 用例 6: $N 个真正重叠的并发实例全部成功，且 agent 目录互不相同 =="
# `omp bash -c ... sleep` keeps each instance alive for the whole window, so the
# $N instances really do overlap -- this is the invariant that matters: no two
# live instances may share one agent directory (and therefore one SQLite file).
hold_lock 45
sleep 1
pids=""
i=1
while [ "$i" -le "$N" ]; do
  ( "$WRAPPER" "$NS" bash -c 'echo "AGENTDIR=$PI_CODING_AGENT_DIR"; sleep 6' \
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
if grep -q "^$NSSLOTS/" "$TMP/dirs" && ! grep -qx "$CANONICAL" "$TMP/dirs"; then
  ok "所有并发实例都落在 slot 里，没有碰规范目录"
else
  bad "并发实例落到了规范目录，槽位隔离失效"
fi
if grep -qv "^$OMP_HOME/" "$TMP/dirs"; then
  bad "有 agent 目录不在 \$HOME 下: $(grep -v "^$OMP_HOME/" "$TMP/dirs" | head -3)"
else
  ok "所有 agent 目录都在 \$HOME（$OMP_HOME）之下"
fi

echo "== 用例 7: 槽位与规范目录共享配置（软链，不是复制）=="
slot=$(head -1 "$TMP/dirs")
if [ -L "$slot/config.yml" ] && [ "$(readlink "$slot/config.yml")" = "$CANONICAL/config.yml" ]; then
  ok "槽位 config.yml 软链到规范目录"
else
  bad "槽位 config.yml 不是指向规范目录的软链: $(ls -l "$slot/config.yml" 2>&1)"
fi
hold_lock 8
sleep 1
listing=$("$WRAPPER" "$NS" config list --scope=global 2>/dev/null)
release_lock
marker=$(grep -m1 -A1 'modelRoles' "$CANONICAL/config.yml" | tail -1 | tr -d ' ' | cut -d: -f2-)
if [ -n "$marker" ] && case "$listing" in *"$marker"*) true ;; *) false ;; esac; then
  ok "槽位实例读到共享配置（$marker）"
else
  bad "槽位实例没有读到共享配置"
fi

echo "== 用例 8: SQLite 文件不被两个实例共用 =="
if [ -f "$CANONICAL/agent.db" ] && [ -f "$slot/agent.db" ]; then
  i1=$(ls -i "$CANONICAL/agent.db" | cut -d' ' -f1)
  i2=$(ls -i "$slot/agent.db" | cut -d' ' -f1)
  if [ "$i1" != "$i2" ]; then
    ok "规范目录与槽位的 agent.db 是不同 inode（$i1 vs $i2）"
  else
    bad "两个实例共用同一个 agent.db"
  fi
else
  bad "缺少 agent.db，无法校验隔离"
fi

echo "== 用例 9: 存活实例持有的槽位不会被抢占 =="
mkdir -p "$NSSLOTS/live-owner" "$SLOTLOCKS"
hold_slot_lock live-owner 8
sleep 1
hold_lock 8
sleep 1
d=$(agent_dir_of "$NS")
release_lock
case "$d" in
  *live-owner*) bad "抢占了存活实例持有的槽位: $d" ;;
  *) ok "存活实例的槽位未被抢占（选到了 $d）" ;;
esac

echo "== 用例 10: 空闲（陈旧）的槽位被复用，而不是无限堆积 =="
# 先把其它槽位清掉，只留 live-owner 和本用例新建的空闲槽位，
# 这样 wrapper 只有一个可选目标，断言才是确定的。
for d in "$NSSLOTS"/*; do
  case "${d##*/}" in .*|live-owner|free-owner) continue ;; esac
  /bin/rm -rf "$d"
done
for l in "$SLOTLOCKS"/*; do
  case "${l##*/}" in live-owner.lock) continue ;; esac
  /bin/rm -f "$l"
done
mkdir -p "$NSSLOTS/free-owner"
before=$(/bin/ls -1 "$NSSLOTS" 2>/dev/null | wc -l)
hold_lock 8
sleep 1
d=$(agent_dir_of "$NS")
release_lock
case "$d" in
  *free-owner*) ok "空闲槽位被复用: $d" ;;
  *) bad "空闲槽位没有被复用（选到了 $d）" ;;
esac
after=$(/bin/ls -1 "$NSSLOTS" 2>/dev/null | wc -l)
if [ "$after" -le "$before" ]; then
  ok "槽位数量未增长（$before -> $after），没有新造槽位"
else
  bad "槽位数量增长（$before -> $after），没有复用空闲槽位"
fi

echo "== 用例 11: 槽位的归属锁不放在 omp 的 agent 目录里 =="
# omp 自己拥有 agent 目录，会清理它不认识的文件；锁文件必须放在别处，
# 否则锁文件被删后重新创建会换 inode，新来的实例就拿不到同一把锁。
if [ -d "$SLOTLOCKS" ]; then
  ok "归属锁目录在 $SLOTLOCKS（agent 目录之外）"
else
  bad "缺少 $SLOTLOCKS 锁目录"
fi
if find "$NSSLOTS" -mindepth 2 -maxdepth 2 -name '*.lock' -not -path "$SLOTLOCKS/*" 2>/dev/null | grep -q .; then
  bad "有 .lock 文件落在 agent 目录内: $(find "$NSSLOTS" -mindepth 2 -maxdepth 2 -name '*.lock' -not -path "$SLOTLOCKS/*" | head -3)"
else
  ok "agent 目录内没有 .lock 文件"
fi

echo "== 用例 12: 活着的实例确实持有自己的槽位锁 =="
# 挡住规范目录，逼这个实例进 slot，然后验证它真的持有那把槽位锁
# （锁必须跟着进程活过 exec，否则别的实例就能抢走同一个槽位）。
hold_lock 30
( "$WRAPPER" "$NS" bash -c 'echo "AGENTDIR=$PI_CODING_AGENT_DIR"; sleep 10' \
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

echo "== 用例 13: 不同命名空间的实例读到同一份配置 =="
# 这是"所有实例应该使用一样的"这条要求的直接检验。
hold_lock 20
sleep 1
d1=$("$WRAPPER" nsa config path 2>/dev/null | grep -E '^/' | tail -1)
d2=$("$WRAPPER" nsb config path 2>/dev/null | grep -E '^/' | tail -1)
release_lock
if [ -n "$d1" ] && [ -n "$d2" ] && [ "$d1" != "$d2" ]; then
  ok "两个命名空间拿到了不同的 agent 目录: $d1 / $d2"
else
  bad "两个命名空间的 agent 目录相同: $d1 / $d2"
fi
if [ -f "$d1/config.yml" ] && [ -f "$d2/config.yml" ] &&
   [ "$(sha256sum "$d1/config.yml" | cut -d' ' -f1)" = "$(sha256sum "$d2/config.yml" | cut -d' ' -f1)" ] &&
   [ "$(sha256sum "$d1/config.yml" | cut -d' ' -f1)" = "$(sha256sum "$CANONICAL/config.yml" | cut -d' ' -f1)" ]; then
  ok "两个命名空间读到的配置与规范目录完全相同"
else
  bad "不同命名空间读到的配置不一致"
fi
if [ -L "$d1/config.yml" ] && [ "$(readlink "$d1/config.yml")" = "$CANONICAL/config.yml" ] &&
   [ -L "$d2/config.yml" ] && [ "$(readlink "$d2/config.yml")" = "$CANONICAL/config.yml" ]; then
  ok "两个命名空间都软链到同一份配置"
else
  bad "命名空间没有软链到同一份配置"
fi
for d in "$d1" "$d2"; do
  case "$d" in
    "$OMP_HOME"/profiles/*) bad "实例仍落在 profile 树里: $d" ;;
    *) ok "实例不在 profile 树里: $d" ;;
  esac
done

echo "== 用例 14: 槽位拿到与规范实例相同的凭据 =="
# 凭据存在规范实例的 agent.db 里。槽位有自己的数据库，必须把规范实例的库
# 整个复制过去：只复制凭据那张表是不行的，那样槽位的库是残缺的 schema，
# omp 会把它改名成 ".corrupt-*" 并重建 -- 而改名会 unlink 文件，使进程
# 自己的映射失效，omp 于是死于 SIGBUS。
hold_lock 8
sleep 1
s2=$("$WRAPPER" "$NS" config path 2>/dev/null | grep -E '^/' | tail -1)
release_lock
canon_rows=$(rows_in "$CANONICAL/agent.db" auth_credentials)
if [ "$canon_rows" = "-1" ]; then
  skipt "规范实例没有 auth_credentials 表"
else
  slot_rows=$(rows_in "$s2/agent.db" auth_credentials)
  if [ "$slot_rows" = "$canon_rows" ]; then
    ok "槽位的凭据条数与规范实例一致（$slot_rows 条）"
  else
    bad "槽位的凭据条数与规范实例不一致（$slot_rows vs $canon_rows）"
  fi
fi

echo "== 用例 15: 跨节点并发（可选，需要 OMP_TEST_PEER）=="
if [ -z "$PEER" ]; then
  skipt "未设置 OMP_TEST_PEER，跳过跨节点用例"
else
  ( "$WRAPPER" "$NS" bash -c 'echo "AGENTDIR=$PI_CODING_AGENT_DIR"; sleep 12' \
      >"$TMP/out.peer" 2>/dev/null ) &
  PEER_PID=$!
  sleep 5
  peer_out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
    "'$WRAPPER' '$NS' bash -c 'echo PEERDIR=\$PI_CODING_AGENT_DIR; sleep 2'" 2>&1 |
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
