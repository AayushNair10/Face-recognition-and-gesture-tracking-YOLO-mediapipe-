"""Headless check that everything works on this machine (no webcam or screen needed).

  .venv/bin/python smoke_test.py

Runs the face tracker on a sample photo, the gesture model on MediaPipe's thumbs-up photo,
and prints PASS/FAIL per step; exit code 1 if anything failed. Needs no internet.
"""
import platform
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
failed = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failed.append(name)


def main():
    from face_tracker import HeadTracker, pick_device  # first: it switches Ultralytics to offline mode

    import mediapipe as mp
    import torch
    from ultralytics.utils import ASSETS

    from camera import FrameSource
    from gestures import Debouncer, GestureDetector, GestureWorker

    print(f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, torch {torch.__version__}, "
          f"OpenCV {cv2.__version__}, MediaPipe {mp.__version__}, device {pick_device()}\n")
    check("OpenCV has the ViT tracker", hasattr(cv2, "TrackerVit_create"))
    for f in ("yolo26n-pose.pt", "object_tracking_vittrack_2023sep.onnx", "gesture_recognizer.task", "thumbs_up.jpg"):
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
    hand_img = cv2.imread(str(MODELS / "thumbs_up.jpg"))
    h, w = hand_img.shape[:2]
    worker, deb, events, hands, ms, runs = GestureWorker(gd), Debouncer(), [], [], 0.0, 0
    deadline = time.perf_counter() + 30
    while runs < 6 and time.perf_counter() < deadline:  # same path as the app: gestures on their own thread
        if not worker.busy:
            worker.submit(hand_img, [0, 0, w, h])
        done = worker.poll()
        if done:
            hands, ms = done
            runs += 1
            events += deb.update(hands, time.perf_counter())
        time.sleep(0.001)
    label = hands[0].builtin if hands else None
    check("gesture recognised (thumbs up)", label == "Thumb_Up",
          f"got {label} {hands[0].builtin_score:.2f}, {ms:.0f} ms" if hands else "no hand found")
    check("held gesture fires exactly one event", [e.label for e in events] == ["Thumb_Up"], str([e.label for e in events]))
    worker.close()

    # video source
    clip = Path(tempfile.mkdtemp()) / "clip.avi"
    out = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 30, (img.shape[1], img.shape[0]))
    for _ in range(5):
        out.write(img)
    out.release()
    src = FrameSource(str(clip))
    n = 0
    while src.read()[0] is not None:
        n += 1
    src.release()
    check("frame source reads a video", n == 5, f"{n} frames")


if __name__ == "__main__":
    main()
    print("\nALL PASSED" if not failed else f"\n{len(failed)} FAILED: {', '.join(failed)}")
    sys.exit(1 if failed else 0)
