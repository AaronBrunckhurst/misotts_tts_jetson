#!/usr/bin/env bash
# uninstall.sh — Remove MisoTTS and its service from a Jetson
set -euo pipefail

REPO_DIR="$HOME/misotts"
PY_VERSION="3.10.14"
HF_CACHE="$HOME/.cache/huggingface/hub/models--MisoLabs--MisoTTS"
SERVICE_DST="/etc/systemd/system/misotts.service"

echo "=== MisoTTS Uninstaller ==="
echo ""
echo "This will:"
echo "  - Stop and disable the misotts systemd service"
echo "  - Remove $SERVICE_DST"
echo "  - Delete $REPO_DIR (server code, voices, outputs)"
echo ""
read -rp "Continue? [y/N] " CONFIRM
[[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

sudo -v

# ── Service ───────────────────────────────────────────────────────────────────
echo ""
echo "Stopping and disabling misotts service..."
sudo systemctl stop misotts   2>/dev/null || true
sudo systemctl disable misotts 2>/dev/null || true
sudo rm -f "$SERVICE_DST"
sudo systemctl daemon-reload

# ── Repo dir ──────────────────────────────────────────────────────────────────
echo "Removing $REPO_DIR ..."
rm -rf "$REPO_DIR"

# ── Optional: HuggingFace model cache ────────────────────────────────────────
echo ""
if [ -d "$HF_CACHE" ]; then
    read -rp "Also delete the 16 GB HuggingFace model cache at $HF_CACHE? [y/N] " DEL_CACHE
    if [[ "$DEL_CACHE" =~ ^[Yy]$ ]]; then
        rm -rf "$HF_CACHE"
        echo "Cache deleted."
    fi
fi

# ── Optional: pyenv Python 3.10 ───────────────────────────────────────────────
if command -v pyenv &>/dev/null; then
    read -rp "Also remove pyenv Python $PY_VERSION (only safe if nothing else uses it)? [y/N] " DEL_PY
    if [[ "$DEL_PY" =~ ^[Yy]$ ]]; then
        pyenv uninstall -f "$PY_VERSION"
        echo "Python $PY_VERSION removed."
    fi
fi

echo ""
echo "=== Uninstall complete ==="
