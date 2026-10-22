#!/usr/bin/env bash
# One-time setup: venv + packages + model files. Works on macOS (Apple silicon) and Ubuntu (x86_64 / ARM).
set -euo pipefail
cd "$(dirname "$0")"

OS=$(uname -s)
BENCH="$HOME/Downloads/benchmark_face"

# Python: $PY if given, else the newest supported one on PATH
if [ -z "${PY:-}" ]; then
  for c in python3.13 python3.12 /opt/homebrew/bin/python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi
  done
fi
PYV=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "using $PY (Python $PYV) on $OS $(uname -m)"

if [ "$OS" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
  # portaudio: mediapipe imports sounddevice at startup | egl/gles: mediapipe's native library
  # libgl/glib/xcb: OpenCV window
  DEPS="libportaudio2 libegl1 libgles2 libgl1 libglib2.0-0 libxcb-xinerama0 libxkbcommon-x11-0 curl"
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
# Linux PC without an NVIDIA GPU: the CPU build of PyTorch is ~200 MB instead of ~3 GB of CUDA libraries
if [ "$OS" = "Linux" ] && [ "$(uname -m)" = "x86_64" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  .venv/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -r requirements.txt
# ultralytics pulls in opencv-python, which shadows opencv-contrib-python (same cv2 module) -> keep only contrib
.venv/bin/python -m pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
.venv/bin/python -m pip install --force-reinstall --no-deps -q opencv-contrib-python==5.0.0.93

mkdir -p models
fetch() {  # fetch <dest> <local copy> <url>
  if [ -s "models/$1" ]; then return; fi
  if [ -s "$2" ]; then cp "$2" "models/$1"; echo "copied $1"
  else echo "downloading $1"; curl -fL --retry 3 -o "models/$1" "$3"; fi
}
fetch yolo26n-pose.pt "$BENCH/yolo26n-pose.pt" \
  https://github.com/ultralytics/assets/releases/latest/download/yolo26n-pose.pt
fetch object_tracking_vittrack_2023sep.onnx "$BENCH/bench_results/weights/object_tracking_vittrack_2023sep.onnx" \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx
fetch gesture_recognizer.task /nonexistent \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task

.venv/bin/python - <<'PY'
import cv2, mediapipe as mp, torch
from mediapipe.tasks.python import vision
from face_tracker import pick_device
print("cv2", cv2.__version__, "| TrackerVit:", hasattr(cv2, "TrackerVit_create"))
print("mediapipe", mp.__version__, "| GestureRecognizer:", hasattr(vision, "GestureRecognizer"))
print("torch", torch.__version__, "| device:", pick_device())
PY
echo "Setup done. Check it:  .venv/bin/python smoke_test.py    Run it:  .venv/bin/python app.py"
