#!/usr/bin/env bash
# YARO 一键安装（macOS / Linux）
#
#   ./scripts/install.sh            # 云 API + 视觉依赖 + 独立窗口
#   ./scripts/install.sh --editable # 开发模式
#   EXTRAS=cv ./scripts/install.sh  # 只要视觉依赖，不要云 API / 不要窗口
#
# 独立窗口（pywebview）解决"每个人浏览器环境不一样"的普适性问题：它用系统 WebView，
# 不受浏览器扩展和强制深色影响。装不上时自动降级为浏览器模式，不影响其余功能。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXTRAS="${EXTRAS:-full}"
EDITABLE=0
SKIP_INIT=0
NO_DESKTOP=0

for arg in "$@"; do
  case "$arg" in
    --editable) EDITABLE=1 ;;
    --skip-init) SKIP_INIT=1 ;;
    --no-desktop) NO_DESKTOP=1 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done

PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "[1/5] 创建虚拟环境 .venv"
  python3 -m venv "$ROOT/.venv"
else
  echo "[1/5] 复用已有 .venv"
fi

echo "[2/5] 升级 pip"
"$PY" -m pip install --upgrade pip --quiet

echo "[3/5] 安装 yaro[$EXTRAS]"
if [ -n "$EXTRAS" ]; then
  if [ "$EDITABLE" = "1" ]; then "$PY" -m pip install -e ".[$EXTRAS]"; else "$PY" -m pip install ".[$EXTRAS]"; fi
else
  if [ "$EDITABLE" = "1" ]; then "$PY" -m pip install -e .; else "$PY" -m pip install .; fi
fi

DESKTOP=0
if [ "$NO_DESKTOP" = "1" ]; then
  echo "[4/5] 跳过独立窗口（--no-desktop）"
elif "$PY" -c "from yaro.desktop import available; raise SystemExit(0 if available() else 1)" 2>/dev/null; then
  DESKTOP=1
  echo "[4/5] 独立窗口可用：用 yaro app 启动（推荐）"
else
  echo "[4/5] 独立窗口不可用：将回退到浏览器模式（yaro gui）"
  echo "      Linux 上通常需要：sudo apt install libwebkit2gtk-4.1-dev"
fi

if [ "$SKIP_INIT" = "0" ]; then
  echo "[5/5] 初始化工作区（幂等）"
  "$PY" -m yaro init
fi

cat <<'EOF'

完成。后续命令：
  source .venv/bin/activate
  yaro doctor
EOF
if [ "$DESKTOP" = "1" ]; then echo "  yaro app      # 独立窗口（推荐）"; else echo "  yaro gui      # 浏览器模式"; fi
