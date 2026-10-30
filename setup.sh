#!/usr/bin/env bash
# One-time setup: venv + packages + model files. Needs internet once; the app itself never does.
# Works on macOS (Apple silicon), Ubuntu (x86_64 / ARM) and NVIDIA Jetson (JetPack 5, 6, 7).
set -euo pipefail
cd "$(dirname "$0")"

OS=$(uname -s)
# Jetson: L4T major version, e.g. 36 from "# R36 (release), REVISION: 4.0, ..."
L4T=$(sed -n 's/^# R\([0-9]*\).*/\1/p' /etc/nv_tegra_release 2>/dev/null || true)

# Python: $PY if given, else the newest supported one on PATH (a Jetson must use its system Python)
[ -n "$L4T" ] && PY=${PY:-python3}
if [ -z "${PY:-}" ]; then
  for c in python3.13 python3.12 /opt/homebrew/bin/python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi
  done
fi
PYV=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "using $PY (Python $PYV) on $OS $(uname -m)"

if [ "$OS" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
  # egl/gles: mediapipe's native library | libgl/glib/xcb: OpenCV window
  DEPS="libegl1 libgles2 libgl1 libglib2.0-0 libxcb-xinerama0 libxkbcommon-x11-0 curl"
  "$PY" -c "import ensurepip" 2>/dev/null || DEPS="$DEPS python$PYV-venv"  # Ubuntu's Python ships without venv
  # (Ubuntu 24.04 renamed some libraries to <name>t64)
  MISSING=$(for p in $DEPS; do dpkg -s "$p" >/dev/null 2>&1 || dpkg -s "${p}t64" >/dev/null 2>&1 || echo "$p"; done | xargs)
  if [ -n "$MISSING" ]; then
    echo "installing system packages: $MISSING"
    sudo apt-get update -qq && sudo apt-get install -y -qq $MISSING
  fi
fi

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
REQS=requirements.txt
if [ -n "$L4T" ]; then
  # Jetson: the normal PyTorch has no Jetson GPU support, so install the build for this JetPack first
  # (versions from https://docs.ultralytics.com/guides/nvidia-jetson/)
  REQS=requirements-jetson.txt
  ASSETS=https://github.com/ultralytics/assets/releases/download/v0.0.0
  case "$L4T" in
    35) echo "Jetson: JetPack 5"
        .venv/bin/python -m pip install "$ASSETS/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl" \
                                        "$ASSETS/torchvision-0.16.2+c6f3977-cp38-cp38-linux_aarch64.whl" ;;
    36) echo "Jetson: JetPack 6"
        if ! dpkg -s cudss >/dev/null 2>&1; then  # torch 2.10 for Jetson needs NVIDIA's cuDSS library
          CUDSS=cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
          curl -fL --retry 3 -o "/tmp/$CUDSS" "https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/$CUDSS"
          sudo dpkg -i "/tmp/$CUDSS"
          sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
          sudo apt-get update -qq && sudo apt-get install -y -qq cudss
        fi
        .venv/bin/python -m pip install "$ASSETS/torch-2.10.0-cp310-cp310-linux_aarch64.whl" \
                                        "$ASSETS/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl" ;;
    38|39) echo "Jetson: JetPack 7"
        .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 ;;
    *)  echo "Jetson with L4T R$L4T is not supported: this needs JetPack 5, 6 or 7 (the original Jetson Nano stops at JetPack 4)."
        exit 1 ;;
  esac
elif [ "$OS" = "Linux" ] && [ "$(uname -m)" = "x86_64" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  # Linux PC without an NVIDIA GPU: the CPU build of PyTorch is ~200 MB instead of ~3 GB of CUDA libraries
  .venv/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -r "$REQS"
# ultralytics pulls in opencv-python, which shadows opencv-contrib-python (same cv2 module) -> keep only contrib
.venv/bin/python -m pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
.venv/bin/python -m pip install --force-reinstall --no-deps -q opencv-contrib-python==5.0.0.93

mkdir -p models
fetch() {  # fetch <dest> <url>
  if [ ! -s "models/$1" ]; then echo "downloading $1"; curl -fL --retry 3 -o "models/$1" "$2"; fi
}
fetch yolo26n-pose.pt \
  https://github.com/ultralytics/assets/releases/latest/download/yolo26n-pose.pt
fetch object_tracking_vittrack_2023sep.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx
fetch gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
fetch thumbs_up.jpg \
  https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg  # photo for smoke_test.py

.venv/bin/python - <<'PY'
import cv2, mediapipe as mp, torch
from mediapipe.tasks.python import vision
from face_tracker import pick_device
print("cv2", cv2.__version__, "| TrackerVit:", hasattr(cv2, "TrackerVit_create"))
print("mediapipe", mp.__version__, "| GestureRecognizer:", hasattr(vision, "GestureRecognizer"))
print("torch", torch.__version__, "| device:", pick_device())
PY
echo "Setup done. Check it:  .venv/bin/python smoke_test.py    Run it:  .venv/bin/python app.py"
