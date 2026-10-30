"""Live test: face tracking (YOLO26n-pose + ViT hybrid) + hand gestures (MediaPipe), with performance analysis.

  .venv/bin/python app.py                      # run with a window
  .venv/bin/python app.py --headless           # run on the robot: no window, nothing saved
  .venv/bin/python app.py --record wave_help   # record a custom gesture: press c to start/stop recording

Keys: q / Esc quit | r force face re-detect | d toggle HUD | c start/stop recording (with --record)
"""
import argparse
import signal
import threading
import time
from pathlib import Path

import cv2

from camera import FrameSource
from face_tracker import HeadTracker
from gestures import HAND_CONNECTIONS, Debouncer, GestureDetector, GestureWorker, normalize_landmarks
from metrics import Metrics

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
CUSTOM = ROOT / "custom_gestures" / "samples.csv"
WIN = "Face + Gesture test"

GREEN, YELLOW, RED, WHITE, GRAY, CYAN = (80, 220, 80), (0, 215, 255), (60, 60, 230), (255, 255, 255), (150, 150, 150), (230, 200, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale=0.55, color=WHITE, thick=1):
    # drop shadow, same thickness (OpenCV 5 renders thicker text wider, so a thick outline would ghost)
    cv2.putText(img, s, (org[0] + 2, org[1] + 2), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_panel(img, lines, org=(10, 10), scale=0.55):
    lh = int(26 * scale / 0.55)
    w = max(cv2.getTextSize(s, FONT, scale, 1)[0][0] for s, _ in lines) + 20
    h = lh * len(lines) + 12
    x, y = org
    roi = img[y:y + h, x:x + w]
    roi[:] = (roi * 0.35).astype(roi.dtype)  # dim only the panel area (cheap)
    for i, (s, c) in enumerate(lines):
        text(img, s, (x + 10, y + lh * (i + 1)), scale, c)


def draw(frame, head, hands, stats, args, show_hud, banner, rec_info):
    h, w = frame.shape[:2]
    if head.head_box is not None:
        x1, y1, x2, y2 = map(int, head.head_box)
        c = GREEN if head.state == "track" else YELLOW
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        text(frame, "FACE", (x1, max(18, y1 - 8)), 0.6, c, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(frame, (cx, cy), c, cv2.MARKER_CROSS, 14, 2)
    else:
        text(frame, "FACE LOST - searching...", (w // 2 - 150, 40), 0.8, RED, 2)
    if show_hud and head.person_box is not None:
        x1, y1, x2, y2 = map(int, head.person_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), GRAY, 1)

    for hd in hands:
        col = CYAN if hd.owned else GRAY
        pts = hd.landmarks_px.astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, WHITE, -1, cv2.LINE_AA)
        lbl = hd.label or "-"
        src = f" (custom {hd.custom_score:.2f})" if hd.custom else (f" {hd.builtin_score:.2f}" if hd.builtin else "")
        tag = f"{hd.side}: {lbl}{src}" + ("" if hd.owned else "  [not tracked person]")
        x, y = pts[:, 0].min(), pts[:, 1].min() - 10
        text(frame, tag, (int(x), int(max(20, y))), 0.6, col, 2)

    if show_hud:
        s = stats
        norm = f"x {head.head_center_norm[0]:+.2f}, y {head.head_center_norm[1]:+.2f}" if head.head_center_norm else "-"
        lines = [
            (f"FPS {s['fps']:d}   processing {s['proc_ms']:.1f} ms (p95 {s['proc_p95']:.1f})   lag {s['lag_ms']:.1f} ms (p95 {s['lag_p95']:.1f})", WHITE),
            (f"pose {s['pose_ms']:.1f} ms | ViT {s['vit_ms']:.1f} ms | gesture {s['gesture_ms']:.1f} ms | detector on {s['det_pct']:.0f}% of frames", WHITE),
            (f"face: {head.state}   centre offset {norm}", GREEN if head.state != "lost" else RED),
        ]
        if rec_info:
            lines.append(rec_info)
        draw_panel(frame, lines)
        text(frame, "q quit | r re-detect | d HUD" + (" | c record" if args.record else ""), (10, h - 12), 0.5, GRAY)

    if banner and time.perf_counter() - banner[1] < 1.5:
        ev = banner[0]
        msg = f"{ev.label.replace('_', ' ').upper()}  ({ev.side} hand)"
        scale = min(1.3, 1.3 * 0.85 * w / cv2.getTextSize(msg, FONT, 1.3, 3)[0][0])  # fit narrow frames
        (tw, th), _ = cv2.getTextSize(msg, FONT, scale, 3)
        x, y = (w - tw) // 2, h - 60
        cv2.rectangle(frame, (x - 20, y - th - 20), (x + tw + 20, y + 20), (40, 40, 40), -1)
        text(frame, msg, (x, y), scale, YELLOW, 3)


def on_result(head, events):
    """Robot hook, called once per frame.

    head.state             "detect" / "track" = person in view, "lost" = nobody
    head.head_center_norm  where the face is: (x, y), each -1..1 from the image centre (use it to turn the head)
    events                 gestures that just fired (each has .label and .side)
    """


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", help="video file to use instead of the camera (for repeatable tests)")
    ap.add_argument("--headless", action="store_true", help="robot mode: no window, 640x480 capture")
    ap.add_argument("--width", type=int, help="capture width (default 1280, or 640 with --headless)")
    ap.add_argument("--height", type=int, help="capture height (default 720, or 480 with --headless)")
    ap.add_argument("--imgsz", type=int, default=416, help="YOLO input size (lower = faster, shorter range)")
    ap.add_argument("--redetect", type=int, default=15, help="re-run the pose detector every N frames")
    ap.add_argument("--gesture-every", type=int, default=2, help="run gesture recognition every N frames")
    ap.add_argument("--pose-model", default=str(MODELS / "yolo26n-pose.pt"), help=".pt, or a TensorRT .engine on Jetson")
    ap.add_argument("--gesture-model", default=str(MODELS / "gesture_recognizer.task"),
                    help="MediaPipe .task (built-in, or one trained with Model Maker)")
    ap.add_argument("--record", metavar="LABEL", help="record samples for a custom gesture (press c to start/stop)")
    ap.add_argument("--no-gestures", action="store_true")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    args = ap.parse_args()
    if args.headless and args.record:
        ap.error("--record needs the window (it uses the c key), so it can't be used with --headless")

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    print("loading models ...")
    tracker = HeadTracker(args.pose_model, str(MODELS / "object_tracking_vittrack_2023sep.onnx"),
                          redetect=args.redetect, imgsz=args.imgsz, device=args.device)
    gd = None if args.no_gestures else GestureDetector(args.gesture_model, CUSTOM)
    worker = GestureWorker(gd) if gd else None
    if gd and len(gd.custom):
        print(f"custom gestures loaded: {gd.custom.labels} ({len(gd.custom)} samples)")
    deb, metrics = Debouncer(), Metrics()
    width = args.width or (640 if args.headless else 1280)
    height = args.height or (480 if args.headless else 720)
    src = FrameSource(args.source or args.camera, width, height)
    print(f"source {src.size[0]}x{src.size[1]}, device {tracker.device}. " + ("Ctrl+C to stop." if args.headless else "Press q to quit."))
    if not args.headless:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    hands, banner, show_hud, recording, n_rec = [], None, True, False, 0
    since_g, stats, t_stats = args.gesture_every, metrics.live(), 0.0
    try:
        while not stop.is_set():
            frame, t_cap = src.read()
            t_start = time.perf_counter()
            if frame is None:
                print("end of video" if src.is_file else "camera stopped")
                break
            frame = cv2.flip(frame, 1)  # selfie view; also what MediaPipe's handedness assumes

            head = tracker.update(frame)

            g_ms, events = 0.0, []
            if worker:
                done = worker.poll()
                if done:
                    hands, g_ms = done
                    events = deb.update(hands, time.perf_counter())
                    for ev in events:
                        banner = (ev, time.perf_counter())
                        print(f"GESTURE {ev.label} ({ev.side} hand, {ev.source})")
                    if recording:
                        own = [hd for hd in hands if hd.owned] or hands
                        if own:
                            gd.custom.add(args.record, normalize_landmarks(own[0].landmarks_px, own[0].side))
                            n_rec += 1
                since_g += 1
                if head.person_box is None and not args.record:  # nobody to take gestures from: save the work
                    hands = []
                    deb.update([], time.perf_counter())
                elif since_g >= args.gesture_every and not worker.busy:
                    worker.submit(frame, head.person_box)
                    since_g = 0

            on_result(head, events)
            if args.headless:
                continue

            t_d = time.perf_counter()
            if t_d - t_stats > 0.5:  # the panel only changes twice a second
                stats, t_stats = metrics.live(), t_d
            rec_info = None
            if args.record:
                rec_info = (f"RECORDING '{args.record}': {n_rec} samples (c to stop)", RED) if recording else \
                           (f"custom gesture '{args.record}': {n_rec} new samples - press c to record", YELLOW)
            draw(frame, head, hands, stats, args, show_hud, banner, rec_info)
            proc_ms = (time.perf_counter() - t_start) * 1000
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF

            metrics.add_frame({"state": head.state, "pose_ms": head.pose_ms, "vit_ms": head.vit_ms, "gesture_ms": g_ms,
                               "proc_ms": proc_ms, "lag_ms": (time.perf_counter() - t_cap) * 1000})

            if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                tracker.force_redetect()
            elif key == ord("d"):
                show_hud = not show_hud
            elif key == ord("c") and args.record and gd:
                recording = not recording
                if not recording:
                    gd.custom.reload()
                    print(f"saved {n_rec} samples for '{args.record}' -> {CUSTOM}; now recognising {gd.custom.labels}")
    finally:
        src.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if worker:
            worker.close()


if __name__ == "__main__":
    main()
