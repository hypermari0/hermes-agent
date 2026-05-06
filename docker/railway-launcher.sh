#!/bin/bash
# Railway entrypoint: runs the gateway, dashboard, and Caddy (basic-auth proxy)
# in a single container. Any one of them exiting takes the container down so
# Railway restarts the whole thing.
set -e

HERMES_HOME="${HERMES_HOME:-/opt/data}"
INSTALL_DIR="/opt/hermes"

# --- Privilege drop (mirrors docker/entrypoint.sh) ---
if [ "$(id -u)" = "0" ]; then
    if [ -n "$HERMES_UID" ] && [ "$HERMES_UID" != "$(id -u hermes)" ]; then
        usermod -u "$HERMES_UID" hermes
    fi
    if [ -n "$HERMES_GID" ] && [ "$HERMES_GID" != "$(id -g hermes)" ]; then
        groupmod -o -g "$HERMES_GID" hermes 2>/dev/null || true
    fi
    actual_hermes_uid=$(id -u hermes)
    if [ "$(stat -c %u "$HERMES_HOME" 2>/dev/null)" != "$actual_hermes_uid" ]; then
        chown -R hermes:hermes "$HERMES_HOME" 2>/dev/null || true
    fi
    exec gosu hermes "$0" "$@"
fi

# --- Running as hermes from here ---
source "${INSTALL_DIR}/.venv/bin/activate"

mkdir -p "$HERMES_HOME"/{cron,sessions,logs,hooks,memories,skills,skins,plans,workspace,home,plugins}

[ -f "$HERMES_HOME/.env" ]       || cp "$INSTALL_DIR/.env.example"        "$HERMES_HOME/.env"
[ -f "$HERMES_HOME/config.yaml" ] || cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
[ -f "$HERMES_HOME/SOUL.md" ]    || cp "$INSTALL_DIR/docker/SOUL.md"      "$HERMES_HOME/SOUL.md"

if [ -d "$INSTALL_DIR/skills" ]; then
    python3 "$INSTALL_DIR/tools/skills_sync.py" || true
fi

# HackClaw: symlink the skill + plugin from the image into the volume.
# The image install (Dockerfile.railway) already pip-installed hackclaw into
# the venv; these symlinks expose it to Hermes' skills/plugins lookup paths.
if [ -d "$INSTALL_DIR/hackclaw" ]; then
    for pair in "skills/hackclaw:skills" "plugin/hackclaw:plugins"; do
        src="$INSTALL_DIR/hackclaw/${pair%%:*}"
        dst="$HERMES_HOME/${pair##*:}/hackclaw"
        if [ -L "$dst" ] || [ ! -e "$dst" ]; then
            ln -snf "$src" "$dst"
        fi
    done
fi

# Auto-enable the composio plugin when an API key is present.
# Idempotent — `hermes plugins enable` is a no-op if already enabled.
if [ -n "$COMPOSIO_API_KEY" ]; then
    hermes plugins enable composio || true
fi

# HackClaw / TAIKAI MCP: idempotently inject the taikai entry into
# config.yaml when the public domain is known. Lets a fresh deploy go from
# zero to functional OAuth without manually editing the volume — see
# tools/mcp_oauth.py (public_callback_url support) and docker/Caddyfile
# (`/oauth/*` route to MCP_OAUTH_CALLBACK_PORT). Only adds; never overwrites
# an existing taikai entry.
RAILWAY_DOMAIN="${RAILWAY_PUBLIC_DOMAIN:-}"
if [ -n "$RAILWAY_DOMAIN" ] && [ -f "$HERMES_HOME/config.yaml" ]; then
    python3 - "$HERMES_HOME/config.yaml" "$RAILWAY_DOMAIN" <<'PY' || true
import sys
import yaml

cfg_path, domain = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f) or {}

mcp = cfg.get("mcp_servers")
if not isinstance(mcp, dict):
    mcp = {}
    cfg["mcp_servers"] = mcp

if "taikai" not in mcp:
    mcp["taikai"] = {
        "url": "https://mcp.taikai.network/mcp",
        "auth": "oauth",
        "oauth": {
            "redirect_port": 9123,
            "public_callback_url": f"https://{domain}/oauth/callback",
        },
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[launcher] injected taikai mcp_server entry "
          f"(public_callback_url=https://{domain}/oauth/callback)")
PY
fi

# --- Config sanity ---
if [ -z "$DASHBOARD_USER" ] || [ -z "$DASHBOARD_PASSWORD_HASH" ]; then
    echo "ERROR: DASHBOARD_USER and DASHBOARD_PASSWORD_HASH must be set."
    echo "Generate a bcrypt hash with:  caddy hash-password"
    exit 1
fi

# --- Start gateway (background) ---
echo "[launcher] starting gateway..."
hermes gateway run &
GATEWAY_PID=$!

# --- Start dashboard (background, loopback-only) ---
echo "[launcher] starting dashboard on 127.0.0.1:9119..."
hermes dashboard --host 127.0.0.1 --port 9119 --no-open &
DASHBOARD_PID=$!

# --- Caddy (foreground) ---
# If gateway or dashboard dies, trap and exit so Railway restarts the container.
trap 'echo "[launcher] child exited — tearing down"; kill $GATEWAY_PID $DASHBOARD_PID 2>/dev/null; exit 1' CHLD

echo "[launcher] starting caddy on :${PORT:-8080}..."
exec caddy run --config "$INSTALL_DIR/docker/Caddyfile" --adapter caddyfile
