#!/usr/bin/env bash
# Install EndoVision AI on helena. Safe to re-run; never uses sudo.
#
#   bash deploy/helena/install.sh [--check-only] [--dry-run] [--skip-weights]
#                                 [--skip-models] [--allow-no-gpu]
#
#   --check-only    only check the system requirements, change nothing
#   --dry-run       run the checks and print the install commands without running them
#   --skip-weights  do not run scripts/download_weights.py
#   --skip-models   do not prefetch the Swin and chat models from Hugging Face
#   --allow-no-gpu  continue without a usable GPU (testing the install only)
#
# Environment: VENV_DIR (default <repo>/.venv), PYTHON (interpreter for the venv).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
VENV=${VENV_DIR:-$REPO/.venv}
VPY=$VENV/bin/python
REQUIREMENTS=$REPO/requirements_web.txt
SWIN_MODEL=microsoft/swin-tiny-patch4-window7-224

CHECK_ONLY=0 DRY_RUN=0 SKIP_WEIGHTS=0 SKIP_MODELS=0 ALLOW_NO_GPU=0
for arg in "$@"; do
  case $arg in
    --check-only) CHECK_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --allow-no-gpu) ALLOW_NO_GPU=1 ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/!s/^# \{0,1\}//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

WARNINGS=0
say()  { printf '\n== %s\n' "$*"; }
ok()   { printf '   ok    %s\n' "$*"; }
warn() { printf '   WARN  %s\n' "$*"; WARNINGS=$((WARNINGS + 1)); }
die()  { printf '   ERROR %s\n' "$*" >&2; exit 1; }
run()  {
  if [ "$DRY_RUN" = 1 ]; then printf '   [dry-run]'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}
python_supported() {
  "$1" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))' >/dev/null 2>&1
}

# --------------------------------------------------------------------- checks
say "Checking the system (repo: $REPO)"
[ -f "$REPO/server.py" ] || die "server.py not found in $REPO"
case "$REPO$VENV" in *" "*) warn "paths with spaces break the systemd unit; move the repo" ;; esac

HAVE_UV=0
if command -v uv >/dev/null 2>&1; then HAVE_UV=1; ok "uv: $(uv --version)"; else ok "uv not found, using python -m venv + pip"; fi

# 3.10 is the version the pins were tested with, so it is preferred.
if [ -n "${PYTHON:-}" ]; then
  python_supported "$PYTHON" || die "PYTHON=$PYTHON is not Python 3.10-3.12"
elif command -v python3.10 >/dev/null 2>&1 && python_supported python3.10; then
  PYTHON=python3.10
elif [ "$HAVE_UV" = 1 ]; then
  PYTHON=3.10   # uv finds or downloads an interpreter
elif command -v python3 >/dev/null 2>&1 && python_supported python3; then
  PYTHON=python3
else
  die "Python 3.10-3.12 needed. Install it (ask the user: sudo apt install python3.10 python3.10-venv) or install uv."
fi
ok "python for the venv: $PYTHON"

# Without uv, Python needs its venv/pip support (a separate package on Debian/Ubuntu).
if [ "$HAVE_UV" = 0 ] && ! "$VPY" -m pip --version >/dev/null 2>&1 &&
   ! "$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1; then
  PYVER=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  hint="$PYTHON cannot create a venv with pip. Ask the user to run: sudo apt install python$PYVER-venv (or install uv)"
  if [ "$CHECK_ONLY" = 1 ] || [ "$DRY_RUN" = 1 ]; then warn "$hint"; else die "$hint"; fi
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
  if [ "${DRIVER%%.*}" -ge 535 ] 2>/dev/null; then ok "NVIDIA driver $DRIVER"; else warn "NVIDIA driver $DRIVER is older than 535; ask the user to update it"; fi
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader | sed 's/^/         GPU: /'
else
  if [ "$ALLOW_NO_GPU" = 1 ] || [ "$CHECK_ONLY" = 1 ] || [ "$DRY_RUN" = 1 ]; then
    warn "nvidia-smi missing or failing: no usable GPU on this machine"
  else
    die "nvidia-smi missing or failing. Fix the driver first (or --allow-no-gpu to test the install only)."
  fi
fi

if command -v ffmpeg >/dev/null 2>&1; then
  ok "ffmpeg: $(command -v ffmpeg)"
else
  warn "no system ffmpeg (the app uses the imageio-ffmpeg binary, verified after the install). To add it: sudo apt install ffmpeg"
fi

# server.py loads exactly these paths for the PDF report.
FONT_DIR=/usr/share/fonts/truetype
if [ -f "$FONT_DIR/dejavu/DejaVuSans.ttf" ] && [ -f "$FONT_DIR/dejavu/DejaVuSans-Bold.ttf" ]; then
  ok "DejaVu fonts"
elif [ -f "$FONT_DIR/liberation/LiberationSans-Regular.ttf" ]; then
  ok "Liberation fonts (fallback for DejaVu)"
else
  warn "DejaVu font missing, PDF reports will look poor. Ask the user to run: sudo apt install fonts-dejavu-core"
fi

free_gb() {  # free space of the filesystem holding $1 (or its closest existing parent)
  local d=$1; while [ ! -e "$d" ]; do d=$(dirname "$d"); done
  df -Pk "$d" | awk 'NR==2 {printf "%d", $4/1024/1024}'
}
for dir in "$REPO" "${HF_HOME:-$HOME/.cache/huggingface}"; do
  GB=$(free_gb "$dir")
  if [ "$GB" -ge 30 ]; then ok "${GB} GB free for $dir"; else warn "only ${GB} GB free for $dir (a fresh install needs about 30 GB in total)"; fi
done

if [ "$CHECK_ONLY" = 1 ]; then say "Check finished with $WARNINGS warning(s)"; exit 0; fi

[ -f "$REQUIREMENTS" ] || die "$REQUIREMENTS not found: is the right branch checked out (duc/pretraining)?"

# --------------------------------------------------------------- environment
say "Python environment: $VENV"
if [ -x "$VPY" ]; then
  python_supported "$VPY" || die "existing venv has an unsupported Python; remove $VENV and re-run"
  ok "reusing the existing venv"
elif [ "$HAVE_UV" = 1 ]; then
  run uv venv --python "$PYTHON" "$VENV"
else
  existed=0; if [ -e "$VENV" ]; then existed=1; fi
  if ! run "$PYTHON" -m venv "$VENV"; then
    # Do not leave a half-made venv behind that the next run would try to reuse.
    if [ "$existed" = 0 ]; then rm -rf -- "$VENV"; fi
    die "venv creation failed. Ask the user to run: sudo apt install python3-venv"
  fi
fi

# requirements_web.txt pins every version, including the CUDA 12.1 torch wheels and
# the extra index they come from.
say "Installing $(basename "$REQUIREMENTS") (torch cu121 and the rest, about 6 GB)"
if [ "$HAVE_UV" = 1 ]; then
  # uv otherwise takes a package only from the first index that has its name.
  run uv pip install --python "$VPY" --index-strategy unsafe-best-match -r "$REQUIREMENTS"
else
  # A venv created earlier by uv has no pip.
  if [ "$DRY_RUN" = 0 ] && ! "$VPY" -m pip --version >/dev/null 2>&1; then run "$VPY" -m ensurepip --upgrade; fi
  run "$VPY" -m pip install -r "$REQUIREMENTS"
fi

# ---------------------------------------------------------------------- .env
say "Configuration file $REPO/.env"
if [ -f "$REPO/.env" ]; then
  ok ".env exists, left unchanged"
else
  run install -m 600 "$SCRIPT_DIR/.env.example" "$REPO/.env"
  if [ "$DRY_RUN" = 0 ]; then ok ".env created from .env.example: APP_USER and APP_PASSWORD still have to be set"; fi
fi
run chmod 600 "$REPO/.env"

env_value() {  # read one non-secret setting from .env without sourcing the file
  [ -f "$REPO/.env" ] || return 0
  sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$REPO/.env" | tail -n1 |
    sed -e 's/[[:space:]]*$//' -e "s/^'\(.*\)'\$/\1/" -e 's/^"\(.*\)"$/\1/'
}
# Process environment first, then .env, like server.py.
for key in HF_HOME WEIGHTS_DIR CHAT_MODEL_ID MULTITASK_REPO; do
  if [ -z "${!key:-}" ]; then
    value=$(env_value "$key")
    case $value in "~/"*) value=$HOME/${value#"~/"} ;; esac
    if [ -n "$value" ]; then export "$key=$value"; fi
  fi
done
CHAT_MODEL=${CHAT_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}

# ------------------------------------------------------------------ downloads
if [ "$SKIP_WEIGHTS" = 1 ]; then
  say "Skipping the weights download"
else
  say "Downloading the model weights (about 5.5 GB; skipped if already complete)"
  [ -f "$REPO/scripts/download_weights.py" ] || die "scripts/download_weights.py not found"
  (cd "$REPO" && run "$VPY" scripts/download_weights.py) || die "weights download failed (see DEPLOY_HELENA.md, troubleshooting)"
fi

if [ "$SKIP_MODELS" = 1 ]; then
  say "Skipping the Hugging Face model prefetch"
else
  say "Prefetching $SWIN_MODEL and $CHAT_MODEL (about 15 GB)"
  run "$VPY" -c '
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors"])
snapshot_download(sys.argv[2], allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("model cache ready")
' "$SWIN_MODEL" "$CHAT_MODEL" || die "model prefetch failed (see DEPLOY_HELENA.md, troubleshooting)"
fi

# --------------------------------------------------------------------- verify
say "Verifying the installation"
if [ "$DRY_RUN" = 1 ]; then
  ok "skipped in dry-run mode"
else
  ALLOW_NO_GPU=$ALLOW_NO_GPU "$VPY" - <<'PY' || die "verification failed (see DEPLOY_HELENA.md, troubleshooting)"
import os, subprocess, sys
import torch, torchvision, transformers, accelerate, bitsandbytes, fastapi, uvicorn, imageio_ffmpeg, cv2, timm
import peft, sklearn, itsdangerous, multipart, dotenv
print(f"   ok    torch {torch.__version__}, transformers {transformers.__version__}, "
      f"bitsandbytes {bitsandbytes.__version__}")
encoders = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-encoders"],
                          capture_output=True, text=True, check=True).stdout
if "libx264" not in encoders:
    sys.exit("   ERROR the ffmpeg used by imageio-ffmpeg has no libx264 encoder")
print("   ok    ffmpeg with libx264")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"   ok    CUDA: {p.name}, {p.total_memory / 1024**3:.1f} GB")
elif os.environ.get("ALLOW_NO_GPU") == "1":
    print("   WARN  torch sees no GPU")
else:
    sys.exit("   ERROR torch.cuda.is_available() is False")
PY
fi

# ----------------------------------------------------------------- next steps
say "Done with $WARNINGS warning(s). Next steps (details: deploy/helena/DEPLOY_HELENA.md)"
cat <<NEXT

1. Set APP_USER and APP_PASSWORD in $REPO/.env (never print or commit it).
2. Smoke test (own endoscopy image):
     cd $REPO && $VPY scripts/smoke_test_webapp.py --image /path/to/frame.jpg
3. The commands below need root. Agents: show them to the user and ask before running.

   # systemd service
   sed -e 's|__USER__|$(id -un)|g' -e 's|__REPO__|$REPO|g' -e 's|__VENV__|$VENV|g' \\
       $SCRIPT_DIR/endovision.service | sudo tee /etc/systemd/system/endovision.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now endovision

   # Caddy (HTTPS): install Caddy, merge deploy/helena/Caddyfile.example with your
   # domain into /etc/caddy/Caddyfile, then
   sudo caddy validate --config /etc/caddy/Caddyfile
   sudo systemctl reload caddy

   # firewall: only 80/443 from outside (keep SSH allowed or you lock yourself out)
   sudo ufw status verbose
   sudo ufw allow OpenSSH
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
NEXT
