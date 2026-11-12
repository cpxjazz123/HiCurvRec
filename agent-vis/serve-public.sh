#!/usr/bin/env bash
# 启动 agent-vis 的公网入口：本地服务 + Cloudflare 隧道，并把公开地址打出来。
#
#   ./serve-public.sh start    启动（或复用已在跑的服务），打印公开地址
#   ./serve-public.sh status   只看当前状态
#   ./serve-public.sh stop     停掉隧道与服务
#
# 公开地址是「Cloudflare 域名 + 访问门槛路径」：
#   域名由 Cloudflare 隧道随机分配（每次重启都会变）；
#   路径来自 .local-public-path —— 服务端的访问门槛，裸路径一律 404。
# 这个服务能调真实模型、能读仓库里的文件，所以门槛不能去掉。
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR=".out"
SERVICE_LOG="$LOG_DIR/service.log"
TUNNEL_LOG="$LOG_DIR/tunnel.log"
PORT=8123
CLOUDFLARED="${CLOUDFLARED:-$(command -v cloudflared || echo /home/wlia0047/bin/cloudflared)}"

mkdir -p "$LOG_DIR"

public_path() {
  if [ -f .local-public-path ]; then
    tr -d '[:space:]' < .local-public-path
  else
    echo ""
  fi
}

service_pid() {
  # 用 ^ 锚定命令行开头，免得把"命令行里恰好提到 server.js 的 bash"也当成服务。
  pgrep -f "^node .*agent-vis/server\.js" | head -1 || true
}

tunnel_pid() {
  # 同理：只认真正跑 cloudflared 的那个进程，不认包装它的 shell。
  pgrep -f "^${CLOUDFLARED} tunnel --url" | head -1 || true
}

public_domain() {
  # 从隧道日志里找已经分配到的域名。
  grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$TUNNEL_LOG" 2>/dev/null | head -1 || true
}

# 域名在 Cloudflare 侧注册完不等于本机已经能解析 —— DNS 记录要传播一会儿。
# 打印地址之前先确认它真能解析，否则给出的链接当场点不开。
domain_resolves() {
  local host="$1"
  [ -n "$host" ] || return 1
  getent hosts "$host" > /dev/null 2>&1 || nslookup "$host" > /dev/null 2>&1
}

start() {
  if [ ! -f .local-public-path ]; then
    echo "缺少 .local-public-path：先写一段访问门槛路径进去（内容是一段路径，如 vis-xxxxxxxx）。" >&2
    exit 1
  fi

  if [ -z "$(service_pid)" ]; then
    echo "启动本地服务 (127.0.0.1:$PORT) ..."
    nohup node "$PWD/server.js" >> "$SERVICE_LOG" 2>&1 &
    disown || true
    for _ in $(seq 1 30); do
      sleep 0.5
      curl -sf -o /dev/null "http://127.0.0.1:$PORT/$(public_path)/api/health" && break
    done
  else
    echo "本地服务已在跑 (pid $(service_pid))"
  fi

  if [ -z "$(tunnel_pid)" ]; then
    echo "启动 Cloudflare 隧道 ..."
    # 旧日志里的域名是上一条隧道的 —— 留着它，"分配到域名了没有"这个判断
    # 会立刻成真，打印出来的却是已经失效的旧地址。先把旧日志挪走。
    if [ -f "$TUNNEL_LOG" ]; then
      mv "$TUNNEL_LOG" "$TUNNEL_LOG.old"
    fi
    nohup "$CLOUDFLARED" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate >> "$TUNNEL_LOG" 2>&1 &
    disown || true
    for _ in $(seq 1 60); do
      sleep 1
      domain="$(public_domain)"
      if [ -n "$domain" ] && domain_resolves "${domain#https://}"; then
        break
      fi
    done
  else
    echo "隧道已在跑 (pid $(tunnel_pid))"
  fi

  local domain path
  domain="$(public_domain)"
  path="$(public_path)"
  echo
  echo "公开地址: ${domain}/${path}/"
  echo "裸路径会被挡下（404）：${domain}/"
  echo
  echo "服务日志: $SERVICE_LOG    隧道日志: $TUNNEL_LOG"
}

status() {
  echo "服务 pid: $(service_pid || echo 无)"
  echo "隧道 pid: $(tunnel_pid || echo 无)"
  echo "公开地址: $(public_domain)/$(public_path)/"
}

stop() {
  local sp tp
  sp="$(service_pid || true)"
  tp="$(tunnel_pid || true)"
  [ -n "$tp" ] && kill "$tp" 2>/dev/null || true
  [ -n "$sp" ] && kill "$sp" 2>/dev/null || true
  echo "已停。"
}

case "${1:-start}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  *)
    echo "用法: $0 [start|status|stop]" >&2
    exit 2
    ;;
esac
