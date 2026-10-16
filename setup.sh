#!/usr/bin/env bash
# One-time setup: venv + packages + model files.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/opt/homebrew/bin/python3.13}
BENCH="$HOME/Downloads/benchmark_face"

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
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
print("cv2", cv2.__version__, "| TrackerVit:", hasattr(cv2, "TrackerVit_create"))
print("mediapipe", mp.__version__, "| GestureRecognizer:", hasattr(vision, "GestureRecognizer"))
print("torch", torch.__version__, "| MPS:", torch.backends.mps.is_available())
PY
echo "Setup done. Run:  .venv/bin/python app.py"
