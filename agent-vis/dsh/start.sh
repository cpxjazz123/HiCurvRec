#!/usr/bin/env bash
# 以 dsh profile 的方式启动 agent-vis。
#
#   ./dsh/start.sh                 # 前台启动，HTTP 在 config.js 的 HOST:PORT
#   DSH_HOME=/some/home ./dsh/start.sh
#
# 做三件事：
#   1. 找一个 Node >= 24（dsh 0.2.2 要求 >= 24.9）；$DSH_NODE 可显式指定；
#   2. 把 commandcode 的 key 从 ~/.config/opencode/commandcode.key 读进
#      COMMANDCODE_API_KEY（只在进程环境里，不落盘、不打印）；
#   3. 首次运行时生成 $DSH_HOME/profiles/agent-vis，然后 exec dsh。
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
REPO=$(cd "$ROOT/.." && pwd)

# 1) Node >= 24
node_ok() {
  local bin=$1
  [[ -x "$bin" ]] || return 1
  local major
  major=$("$bin" -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)
  [[ "$major" -ge 24 ]]
}

NODE_BIN=${DSH_NODE:-}
if [[ -z "$NODE_BIN" ]]; then
  if command -v node >/dev/null 2>&1 && node_ok "$(command -v node)"; then
    NODE_BIN=$(command -v node)
  else
    NODE_BIN=$(find "$HOME/.nvm/versions/node" -maxdepth 3 -type f -name node 2>/dev/null \
      | grep '/v24\.' | sort -V | tail -1)
  fi
fi
if [[ -z "${NODE_BIN:-}" ]] || ! node_ok "$NODE_BIN"; then
  echo "agent-vis: 需要 Node >= 24（dsh 要求 >= 24.9）；用 DSH_NODE=/path/to/node 指定" >&2
  exit 1
fi

# 2) 模型凭证（只进进程环境）
KEY_FILE=${COMMANDCODE_KEY_FILE:-"$HOME/.config/opencode/commandcode.key"}
if [[ -z "${COMMANDCODE_API_KEY:-}" && -r "$KEY_FILE" ]]; then
  COMMANDCODE_API_KEY=$(tr -d '\r\n' <"$KEY_FILE")
  export COMMANDCODE_API_KEY
fi
if [[ -z "${COMMANDCODE_API_KEY:-}" ]]; then
  echo "agent-vis: 没有 COMMANDCODE_API_KEY，也读不到 $KEY_FILE" >&2
  exit 1
fi

# 3) profile + 启动
export DSH_HOME=${DSH_HOME:-"$ROOT/.data/dsh-home"}
if [[ ! -f "$DSH_HOME/profiles/agent-vis/cordis.patch.yml" ]]; then
  "$NODE_BIN" "$HERE/install.mjs" --home "$DSH_HOME"
fi

cd "$REPO"
exec "$NODE_BIN" "$ROOT/node_modules/@deepseek-ai/dsh/lib/bin.js" --profile agent-vis "$@"
