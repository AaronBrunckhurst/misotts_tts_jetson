#!/usr/bin/env bash
# install.sh — Idempotent MisoTTS setup for Jetson AGX Orin (JetPack 5 / Ubuntu 20.04 focal / aarch64)
# Run as the target user (pineway). Prompts for sudo once at the start.
set -euo pipefail

REPO_DIR="$HOME/misotts"
VENV="$REPO_DIR/.venv"
PY_VERSION="3.10.14"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== MisoTTS Jetson Installer ==="
echo "Target: $REPO_DIR"
echo ""

# ── sudo credentials ─────────────────────────────────────────────────────────
echo "This script needs sudo for: apt packages, systemd service install."
sudo -v   # cache credentials; keep alive via background loop
( while true; do sudo -n true; sleep 50; done ) &
SUDO_KEEP_ALIVE=$!
trap 'kill $SUDO_KEEP_ALIVE 2>/dev/null' EXIT

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/13] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git ffmpeg libsndfile1 portaudio19-dev \
    build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev libffi-dev liblzma-dev \
    curl wget

# ── 2. pyenv ─────────────────────────────────────────────────────────────────
echo ""
echo "[2/13] Installing pyenv..."
if [ ! -d "$HOME/.pyenv" ]; then
    curl -fsSL https://pyenv.run | bash
fi

# Ensure pyenv is on PATH for this script session
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Add to ~/.bashrc if not already there
if ! grep -q 'PYENV_ROOT' "$HOME/.bashrc"; then
    cat >> "$HOME/.bashrc" <<'EOF'

# pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
fi

# ── 3. Python 3.10 ───────────────────────────────────────────────────────────
echo ""
echo "[3/13] Building Python $PY_VERSION (skip if already built — takes ~20 min first time)..."
# deadsnakes PPA has no aarch64 focal builds; we compile from source via pyenv
pyenv install -s "$PY_VERSION"
pyenv global "$PY_VERSION"
echo "Python: $(python --version)"

# ── 4. Clone MisoTTS ─────────────────────────────────────────────────────────
echo ""
echo "[4/13] Cloning MisoTTS..."
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone https://github.com/MisoLabsAI/MisoTTS.git "$REPO_DIR"
else
    echo "  Already cloned, skipping."
fi

# ── 5. Create venv ───────────────────────────────────────────────────────────
echo ""
echo "[5/13] Creating Python venv..."
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
$PIP install --quiet --upgrade pip

# ── 6. Core PyTorch (CPU-only on JetPack 5) ──────────────────────────────────
echo ""
echo "[6/13] Installing PyTorch 2.4.0 (CPU-only — no CUDA wheels exist for Python 3.10 on JetPack 5)..."
# The standard PyPI aarch64 manylinux wheel is CPU-only (89.8 MB).
# For GPU: upgrade to JetPack 6 (ships official Py3.10 CUDA wheels) or build from source.
$PIP install --quiet torch==2.4.0 torchaudio==2.4.0

# ── 7. ML / HuggingFace stack ────────────────────────────────────────────────
echo ""
echo "[7/13] Installing ML dependencies..."
$PIP install --quiet \
    tokenizers==0.21.0 \
    transformers==4.49.0 \
    huggingface_hub==0.28.1 \
    torchtune==0.4.0 \
    safetensors==0.5.3 \
    numpy \
    einops \
    psutil

# ── 8. moshi (voice codec) — version-pinned workaround ───────────────────────
echo ""
echo "[8/13] Installing moshi + bitsandbytes (workaround for conflicting pins)..."
# moshi==0.2.2 pins bitsandbytes<0.46, but 0.46.0 is the first build with an
# aarch64 wheel. Install moshi --no-deps, then install the correct versions.
$PIP install --quiet moshi==0.2.2 --no-deps
$PIP install --quiet sentencepiece==0.2.0   # moshi requires ==0.2, 0.2.1 breaks it
$PIP install --quiet bitsandbytes==0.46.0   # first aarch64 wheel
$PIP install --quiet safetensors==0.5.3     # moshi requires <0.6

# ── 9. torchao ───────────────────────────────────────────────────────────────
echo ""
echo "[9/13] Installing torchao (pure-Python wheel)..."
$PIP install --quiet torchao==0.9.0

# ── 10. silentcipher (watermarking) ──────────────────────────────────────────
echo ""
echo "[10/13] Installing silentcipher..."
$PIP install --quiet git+https://github.com/SesameAILabs/silentcipher@d46d7d0

# ── 11. HTTP server deps ──────────────────────────────────────────────────────
echo ""
echo "[11/13] Installing FastAPI / uvicorn..."
$PIP install --quiet "fastapi" "uvicorn[standard]" "python-multipart"

# ── 12. Deploy server script ──────────────────────────────────────────────────
echo ""
echo "[12/13] Deploying tts_server.py..."
cp "$PROJECT_ROOT/server/tts_server.py" "$REPO_DIR/tts_server.py"

# Ensure output/voice directories exist
mkdir -p "$REPO_DIR/outputs" "$REPO_DIR/voices"

# ── 13. Systemd service ───────────────────────────────────────────────────────
echo ""
echo "[13/13] Installing systemd service..."

# Find the exact libgomp path inside the venv
LIBGOMP=$(find "$VENV/lib" -name "libgomp-*.so*" 2>/dev/null | head -1)
if [ -z "$LIBGOMP" ]; then
    echo "WARNING: libgomp not found in venv — LD_PRELOAD will be empty."
    echo "         Run: pip install scikit-learn inside the venv to fix this."
fi

# Render the service file with concrete paths
PYENV_BIN="$HOME/.pyenv/versions/$PY_VERSION/bin"
SERVICE_SRC="$PROJECT_ROOT/systemd/misotts.service"
SERVICE_DST="/etc/systemd/system/misotts.service"
USERNAME="$(whoami)"

# Substitute placeholders
sed \
    -e "s|__USER__|$USERNAME|g" \
    -e "s|__REPO_DIR__|$REPO_DIR|g" \
    -e "s|__VENV__|$VENV|g" \
    -e "s|__LIBGOMP__|$LIBGOMP|g" \
    -e "s|__PYENV_BIN__|$PYENV_BIN|g" \
    "$SERVICE_SRC" | sudo tee "$SERVICE_DST" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable misotts
sudo systemctl restart misotts

echo ""
echo "=== Done! ==="
echo ""
echo "MisoTTS service is running at:  http://$(hostname):8080"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status misotts"
echo "  journalctl -u misotts -f"
echo ""
echo "NOTE: The 16 GB model is NOT loaded at boot (to save memory)."
echo "      Load it from the web UI when you're ready to synthesise."
