"""Headless check that everything works on this machine (no webcam or screen needed).

  .venv/bin/python smoke_test.py

Runs the face tracker on a sample photo, the gesture model on MediaPipe's thumbs-up photo,
and writes a report. Prints PASS/FAIL per step; exit code 1 if anything failed.
"""
import platform
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
THUMBS_UP = "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg"
failed = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failed.append(name)


def main():
    import mediapipe as mp
    import torch
    from ultralytics.utils import ASSETS

    from face_tracker import HeadTracker, pick_device
    from gestures import Debouncer, GestureDetector
    from metrics import Metrics

    print(f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, torch {torch.__version__}, "
          f"OpenCV {cv2.__version__}, MediaPipe {mp.__version__}, device {pick_device()}\n")
    check("OpenCV has the ViT tracker", hasattr(cv2, "TrackerVit_create"))
    for f in ("yolo26n-pose.pt", "object_tracking_vittrack_2023sep.onnx", "gesture_recognizer.task"):
        check(f"model file {f}", (MODELS / f).exists(), "run setup.sh" if not (MODELS / f).exists() else "")
    if failed:
        return

    # face tracking: first frame = YOLO detect, later frames = ViT tracking
    img = cv2.imread(str(ASSETS / "zidane.jpg"))
    tracker = HeadTracker(str(MODELS / "yolo26n-pose.pt"), str(MODELS / "object_tracking_vittrack_2023sep.onnx"))
    res = [tracker.update(img) for _ in range(20)]
    states = [r.state for r in res]
    check("YOLO finds a face", states[0] == "detect", f"{tracker.device}, {res[0].pose_ms:.0f} ms")
    check("ViT tracker follows it", states.count("track") >= 15,
          f"{states.count('track')}/19 tracked, {np.mean([r.vit_ms for r in res if r.state == 'track'] or [0]):.1f} ms")
    check("face box is sensible", all(r.head_box is not None and (r.head_box[2] - r.head_box[0]) > 20 for r in res))

    # gestures
    gd = GestureDetector(MODELS / "gesture_recognizer.task", Path(tempfile.mkdtemp()) / "samples.csv")
    check("MediaPipe runs on an empty frame", gd.process(np.zeros((360, 640, 3), np.uint8)) == [])
    try:
        data = urllib.request.urlopen(THUMBS_UP, timeout=20).read()
        hand_img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except OSError as e:
        print(f"[SKIP] gesture photo download failed ({e})")
        hand_img = None
    if hand_img is not None:
        h, w = hand_img.shape[:2]
        deb, events = Debouncer(), []
        for _ in range(6):
            t = time.perf_counter()
            hands = gd.process(hand_img, [0, 0, w, h])
            ms = (time.perf_counter() - t) * 1000
            events += deb.update(hands, time.perf_counter())
        label = hands[0].builtin if hands else None
        check("gesture recognised (thumbs up)", label == "Thumb_Up",
              f"got {label} {hands[0].builtin_score:.2f}, {ms:.0f} ms" if hands else "no hand found")
        check("held gesture fires exactly one event", [e.label for e in events] == ["Thumb_Up"], str([e.label for e in events]))
    gd.close()

    # report
    m = Metrics()
    for i in range(40):
        time.sleep(0.002)
        m.add_frame({"frame": i, "state": "track", "read_ms": 1, "pose_ms": 0, "vit_ms": 4, "gesture_ms": 0, "draw_ms": 1,
                     "proc_ms": 10, "lag_ms": 12, "gesture_run": 0, "hands": 0, "redetect_iou": None})
    out = Path(tempfile.mkdtemp())
    m.write_report(out, {"test": "smoke"})
    check("session report is written", (out / "summary.md").exists() and (out / "frames.csv").exists())


if __name__ == "__main__":
    main()
    print("\nALL PASSED" if not failed else f"\n{len(failed)} FAILED: {', '.join(failed)}")
    sys.exit(1 if failed else 0)
