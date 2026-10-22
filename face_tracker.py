"""Head tracker: YOLO26n-pose (re-detect) + OpenCV ViT tracker (follow in between).

Same hybrid as the `sot-vit` setup in ~/Downloads/benchmark_face/bench_person_tracking.py, but the
detector step builds a *head* box from the pose keypoints (nose / eyes / ears) instead of using the body box.
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # OpenCV 5 DNN prints a harmless warning per net

# COCO keypoint ids
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO = range(7)


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def clip_box(b, w, h):
    x1, y1, x2, y2 = b
    return [max(0.0, x1), max(0.0, y1), min(w - 1.0, x2), min(h - 1.0, y2)]


def head_box_from_pose(kpts, kconf, person_box, kp_thr=0.5):
    """Square-ish head box from the visible head keypoints; falls back to the top of the person box."""
    vis = [i for i in range(5) if kconf[i] > kp_thr]
    px1, py1, px2, py2 = person_box
    pw = px2 - px1
    if not vis:  # e.g. back of the head: take the top of the body box
        s = 0.35 * pw
        cx = (px1 + px2) / 2
        return [cx - s / 2, py1, cx + s / 2, py1 + 1.2 * s]
    pts = kpts[vis]
    sizes = []
    if kconf[L_EAR] > kp_thr and kconf[R_EAR] > kp_thr:
        sizes.append(1.3 * abs(kpts[L_EAR, 0] - kpts[R_EAR, 0]))
    if len(vis) >= 2:
        sizes.append(1.8 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    if kconf[L_SHO] > kp_thr and kconf[R_SHO] > kp_thr:
        sizes.append(0.45 * abs(kpts[L_SHO, 0] - kpts[R_SHO, 0]))
    s = max(sizes) if sizes else 0.3 * pw
    s = float(np.clip(s, 0.12 * pw, 0.8 * pw)) if pw > 0 else s
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean() - 0.1 * s  # eyes/nose sit below the head centre
    return [cx - s / 2, cy - 0.575 * s, cx + s / 2, cy + 0.575 * s]


def pick_device():
    """NVIDIA GPU (Ubuntu / Jetson) > Apple GPU (Mac) > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class HeadResult:
    head_box: list | None        # xyxy pixels
    person_box: list | None      # xyxy pixels (last detection, shifted with the head)
    head_center_norm: tuple | None  # (-1..1, -1..1) offset from the frame centre - for a pan/tilt robot
    state: str                   # "detect" | "track" | "lost"
    pose_ms: float
    vit_ms: float
    redetect_iou: float | None   # drift check on detect frames (1.0 = ViT was spot on)


class HeadTracker:
    def __init__(self, pose_weights, vit_weights, redetect=15, min_score=0.3, conf=0.25, imgsz=640, device=None):
        from ultralytics import YOLO

        self.device = device or pick_device()
        self.model = YOLO(pose_weights, task="pose")
        params = cv2.TrackerVit_Params()
        params.net = vit_weights
        self.vit = cv2.TrackerVit_create(params)  # created once, re-initialised on every detection
        self.redetect, self.min_score, self.conf, self.imgsz = redetect, min_score, conf, imgsz
        self.sot = None
        self.since_det = 10**9
        self.head_box = self.person_box = None
        self._head_at_det = None
        self.model.predict(np.zeros((360, 640, 3), np.uint8), imgsz=imgsz, device=self.device, verbose=False)  # warm-up

    def force_redetect(self):
        self.since_det = 10**9

    def _detect(self, frame):
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None, None
        boxes = r.boxes.xyxy.cpu().numpy()
        kpts = r.keypoints.xy.cpu().numpy()
        kconf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kpts.shape[:2])
        # same target rule as the benchmark: stay on the last person, else take the largest
        i = max(range(len(boxes)), key=lambda j: (iou(boxes[j], self.person_box), area(boxes[j])))
        person = boxes[i].tolist()
        return head_box_from_pose(kpts[i], kconf[i], person), person

    def update(self, frame):
        h, w = frame.shape[:2]
        pose_ms = vit_ms = 0.0
        box, state, redet_iou = None, "lost", None

        if self.sot is not None and self.since_det < self.redetect:
            t = time.perf_counter()
            ok, (x, y, bw, bh) = self.sot.update(frame)
            score = self.sot.getTrackingScore()
            vit_ms = (time.perf_counter() - t) * 1000
            if ok and score >= self.min_score and bw > 2 and bh > 2:
                box, state = [x, y, x + bw, y + bh], "track"
                self.since_det += 1

        if box is None:  # (re)detect
            t = time.perf_counter()
            head, person = self._detect(frame)
            pose_ms = (time.perf_counter() - t) * 1000
            if head is not None:
                head = clip_box(head, w, h)
                if self.head_box is not None and self.sot is not None:
                    redet_iou = iou(head, self.head_box)
                x1, y1, x2, y2 = head
                self.sot = self.vit
                self.sot.init(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                self.since_det = 0
                self.person_box, self._head_at_det = person, center(head)
                box, state = head, "detect"
            else:
                self.sot, self.person_box = None, None

        self.head_box = box
        person = self.person_box
        if box is not None and person is not None and self._head_at_det is not None:
            # move the (stale) body box along with the head between detections
            cx, cy = center(box)
            dx, dy = cx - self._head_at_det[0], cy - self._head_at_det[1]
            person = [person[0] + dx, person[1] + dy, person[2] + dx, person[3] + dy]
        norm = None
        if box is not None:
            cx, cy = center(box)
            norm = (2 * cx / w - 1, 2 * cy / h - 1)
        return HeadResult(box, person, norm, state, pose_ms, vit_ms, redet_iou)
