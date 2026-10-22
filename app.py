"""Live test: face tracking (YOLO26n-pose + ViT hybrid) + hand gestures (MediaPipe), with performance analysis.

  .venv/bin/python app.py                      # run
  .venv/bin/python app.py --record wave_help   # record a custom gesture: press c to start/stop recording

Keys: q / Esc quit | r force face re-detect | d toggle HUD | c start/stop recording (with --record)
A report table is written to runs/<timestamp>/summary.md (+ frames.csv, events.csv) when you quit.
`.venv/bin/python report.py` compares all runs side by side in runs/all_runs.md.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from face_tracker import HeadTracker
from gestures import HAND_CONNECTIONS, Debouncer, GestureDetector, normalize_landmarks
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
        iou = "n/a" if s["redetect_iou"] != s["redetect_iou"] else f"{s['redetect_iou']:.2f}"
        norm = f"x {head.head_center_norm[0]:+.2f}, y {head.head_center_norm[1]:+.2f}" if head.head_center_norm else "-"
        lines = [
            (f"FPS {s['fps']:d}   processing {s['proc_ms']:.1f} ms (p95 {s['proc_p95']:.1f})   lag {s['lag_ms']:.1f} ms (p95 {s['lag_p95']:.1f})", WHITE),
            (f"pose {s['pose_ms']:.1f} ms | ViT {s['vit_ms']:.1f} ms | gesture {s['gesture_ms']:.1f} ms (every {args.gesture_every}) | draw {s['draw_ms']:.1f} ms", WHITE),
            (f"detector on {s['det_pct']:.0f}% of frames | re-detect IoU {iou} | CPU {s['cpu']:.0f}% | RAM {s['ram_mb']:.0f} MB", WHITE),
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


def open_camera(idx, width, height):
    backend = {"darwin": cv2.CAP_AVFOUNDATION, "linux": cv2.CAP_V4L2}.get(sys.platform, cv2.CAP_ANY)
    cap = cv2.VideoCapture(idx, backend)
    if sys.platform == "linux":  # most USB webcams only reach 30 FPS at 720p in MJPG mode
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(50):  # the first reads can fail while the camera wakes up
        ok, _ = cap.read()
        if ok:
            return cap
        time.sleep(0.05)
    hint = {
        "darwin": "On macOS allow camera access for Terminal / VS Code in System Settings > Privacy & Security > Camera.",
        "linux": f"On Ubuntu check the camera exists (ls /dev/video*), try another --camera index, and make sure your "
                 f"user is in the 'video' group (sudo usermod -aG video $USER, then log out and in).",
    }.get(sys.platform, "Check the camera is connected and not used by another app.")
    raise SystemExit(f"Could not read from camera {idx}. {hint}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--redetect", type=int, default=15, help="re-run the pose detector every N frames")
    ap.add_argument("--gesture-every", type=int, default=2, help="run gesture recognition every N frames")
    ap.add_argument("--gesture-model", default=str(MODELS / "gesture_recognizer.task"),
                    help="MediaPipe .task (built-in, or one trained with Model Maker)")
    ap.add_argument("--record", metavar="LABEL", help="record samples for a custom gesture (press c to start/stop)")
    ap.add_argument("--no-gestures", action="store_true")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    args = ap.parse_args()

    print("loading models ...")
    tracker = HeadTracker(str(MODELS / "yolo26n-pose.pt"), str(MODELS / "object_tracking_vittrack_2023sep.onnx"),
                          redetect=args.redetect, device=args.device)
    gd = None if args.no_gestures else GestureDetector(args.gesture_model, CUSTOM)
    if gd and len(gd.custom):
        print(f"custom gestures loaded: {gd.custom.labels} ({len(gd.custom)} samples)")
    deb, metrics = Debouncer(), Metrics()
    cap = open_camera(args.camera, args.width, args.height)
    print(f"camera {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
          f"device {tracker.device}. Press q to quit.")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    hands, banner, show_hud, recording, n_rec, i = [], None, True, False, 0, 0
    since_g = args.gesture_every
    try:
        while True:
            t_r = time.perf_counter()
            ok, frame = cap.read()
            t_cap = time.perf_counter()
            if not ok:
                print("camera read failed")
                break
            frame = cv2.flip(frame, 1)  # selfie view; also what MediaPipe's handedness assumes

            head = tracker.update(frame)

            # gestures every N frames, nudged by one frame when it would land on a pose re-detect (avoids lag spikes)
            since_g += 1
            g_ms, g_run = 0.0, bool(gd) and since_g >= args.gesture_every and (head.state != "detect" or since_g > args.gesture_every)
            if g_run:
                since_g = 0
                t = time.perf_counter()
                hands = gd.process(frame, head.person_box)
                g_ms = (time.perf_counter() - t) * 1000
                for ev in deb.update(hands, time.perf_counter()):
                    metrics.add_event(ev)
                    banner = (ev, time.perf_counter())
                    print(f"[{time.perf_counter() - metrics.t0:7.2f}s] GESTURE {ev.label} ({ev.side} hand, {ev.source})")
                if recording:
                    own = [hd for hd in hands if hd.owned] or hands
                    if own:
                        gd.custom.add(args.record, normalize_landmarks(own[0].landmarks_px, own[0].side))
                        n_rec += 1

            t_d = time.perf_counter()
            rec_info = None
            if args.record:
                rec_info = (f"RECORDING '{args.record}': {n_rec} samples (c to stop)", RED) if recording else \
                           (f"custom gesture '{args.record}': {n_rec} new samples - press c to record", YELLOW)
            draw(frame, head, hands, metrics.live(), args, show_hud, banner, rec_info)
            draw_ms = (time.perf_counter() - t_d) * 1000
            proc_ms = (time.perf_counter() - t_cap) * 1000

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            lag_ms = (time.perf_counter() - t_cap) * 1000

            metrics.add_frame({
                "frame": i, "state": head.state, "read_ms": round((t_cap - t_r) * 1000, 2),
                "pose_ms": round(head.pose_ms, 2), "vit_ms": round(head.vit_ms, 2), "gesture_ms": round(g_ms, 2),
                "draw_ms": round(draw_ms, 2), "proc_ms": round(proc_ms, 2), "lag_ms": round(lag_ms, 2),
                "gesture_run": int(g_run), "hands": len(hands),
                "redetect_iou": None if head.redetect_iou is None else round(head.redetect_iou, 3)})

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
            i += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if gd:
            gd.close()
        run_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg = {"device": tracker.device, "redetect": args.redetect, "gesture_every": args.gesture_every,
               "camera": f"{args.width}x{args.height}", "gesture_model": Path(args.gesture_model).name,
               "custom_labels": gd.custom.labels if gd else []}
        print("\n" + metrics.write_report(run_dir, cfg))
        print(f"report saved to {run_dir}/")


if __name__ == "__main__":
    main()
