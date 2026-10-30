"""Hand gestures: MediaPipe GestureRecognizer (built-in gestures) + optional custom k-NN on the same hand landmarks.

Custom gestures reuse MediaPipe's hand-landmark model (stage 1) and only swap the classifier (stage 2):
samples recorded with `app.py --record LABEL` go to custom_gestures/samples.csv and are matched with k-NN.
A Model Maker .task file trained in Colab can be dropped in instead with `app.py --gesture-model PATH`.
"""
from __future__ import annotations

import csv
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


@dataclass
class Hand:
    side: str                 # "Left" / "Right" (correct because the frame is mirrored)
    landmarks_px: np.ndarray  # 21 x 2, pixels in the full frame
    builtin: str | None       # MediaPipe label, e.g. "Thumb_Up"
    builtin_score: float
    custom: str | None = None
    custom_score: float = 0.0
    owned: bool = True        # belongs to the tracked person

    @property
    def label(self):
        return self.custom or self.builtin


@dataclass
class GestureEvent:
    t: float
    side: str
    label: str
    source: str  # "builtin" | "custom"


def normalize_landmarks(lm_px, side):
    """Wrist-relative, scale-free, right-hand-canonical 42-d vector (orientation kept: thumb up != thumb down)."""
    p = lm_px - lm_px[0]
    if side == "Left":
        p[:, 0] = -p[:, 0]
    s = np.linalg.norm(p[9]) or 1.0  # wrist -> middle-finger MCP
    return (p / s).ravel()


class CustomGestures:
    """k-NN over normalised landmarks with a distance threshold that rejects unknown hand shapes."""

    def __init__(self, path, k=5):
        self.path, self.k = Path(path), k
        self.reload()

    def reload(self):
        X, y = np.zeros((0, 42), np.float32), np.array([])
        if self.path.exists():
            with open(self.path) as f:
                rows = [r for r in csv.reader(f) if r]
            if rows:
                y = np.array([r[0] for r in rows])
                X = np.array([[float(v) for v in r[1:]] for r in rows], np.float32)
        self._data = (X, y, self._fit_threshold(X, y))  # swapped in one step: the gesture thread may be reading it

    @staticmethod
    def _fit_threshold(X, y):
        if len(X) < 2:
            return 0.0
        nn = []  # distance from each sample to its closest neighbour of the same gesture
        for label in set(y.tolist()):
            same = X[y == label]
            for i in range(len(same) if len(same) > 1 else 0):
                d = np.linalg.norm(same - same[i], axis=1)
                d[i] = np.inf
                nn.append(d.min())
        return max(1.0, 3.0 * float(np.median(nn))) if nn else 1.0

    @property
    def threshold(self):
        return self._data[2]

    @property
    def labels(self):
        return sorted(set(self._data[1].tolist()))

    def __len__(self):
        return len(self._data[1])

    def predict(self, vec):
        X, y, threshold = self._data
        if len(y) < self.k:
            return None, 0.0
        d = np.linalg.norm(X - vec.astype(np.float32), axis=1)
        idx = np.argpartition(d, self.k - 1)[: self.k]
        label, votes = Counter(y[idx].tolist()).most_common(1)[0]
        dist = np.mean([d[i] for i in idx if y[i] == label])
        if votes < (self.k // 2 + 1) or dist > threshold:
            return None, 0.0
        return label, votes / self.k

    def add(self, label, vec):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([label] + [f"{v:.5f}" for v in vec])


class GestureDetector:
    def __init__(self, model_path, custom_path, num_hands=2, min_score=0.5, infer_width=640):
        opts = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands)
        self.rec = vision.GestureRecognizer.create_from_options(opts)
        self.custom = CustomGestures(custom_path)
        self.min_score, self.infer_width = min_score, infer_width
        self._last_ts = -1

    def prepare(self, frame_bgr):
        """Small RGB copy of the frame for MediaPipe (always a new image, so the caller can keep drawing on theirs)."""
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (self.infer_width, int(h * self.infer_width / w))) if w > self.infer_width else frame_bgr
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def process(self, frame_bgr, person_box=None):
        h, w = frame_bgr.shape[:2]
        return self.detect(self.prepare(frame_bgr), w, h, person_box)

    def detect(self, rgb, w, h, person_box=None):
        """Runs the model. w, h: size of the full frame the landmarks are mapped back to."""
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        res = self.rec.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        hands = []
        for i, lms in enumerate(res.hand_landmarks):
            side = res.handedness[i][0].category_name if res.handedness and res.handedness[i] else "?"
            g = res.gestures[i][0] if res.gestures and res.gestures[i] else None
            builtin = g.category_name if g and g.category_name not in ("None", "") and g.score >= self.min_score else None
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
            hand = Hand(side, pts, builtin, g.score if g else 0.0)
            hand.custom, hand.custom_score = self.custom.predict(normalize_landmarks(pts, side))
            hand.owned = _inside(pts[0], person_box)
            hands.append(hand)
        return hands

    def close(self):
        self.rec.close()


class GestureWorker:
    """Runs gesture recognition on its own thread, so the slowest model never delays face tracking."""

    def __init__(self, detector):
        self.detector = detector
        self._job, self._result, self._error, self._stop = None, None, None, False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def busy(self):
        return self._job is not None

    def submit(self, frame_bgr, person_box):
        h, w = frame_bgr.shape[:2]
        job = (self.detector.prepare(frame_bgr), w, h, person_box)
        with self._cond:
            self._job = job
            self._cond.notify()

    def poll(self):
        """(hands, milliseconds) when a result is ready, else None."""
        with self._cond:
            if self._error:
                raise self._error
            result, self._result = self._result, None
        return result

    def _run(self):
        while True:
            with self._cond:
                while self._job is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                job = self._job
            t = time.perf_counter()
            try:
                result, error = (self.detector.detect(*job), (time.perf_counter() - t) * 1000), None
            except Exception as e:  # handed to the main thread by poll()
                result, error = None, e
            with self._cond:
                self._result, self._error, self._job = result, error, None

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=2.0)
        self.detector.close()


def _inside(pt, box, grow_x=0.5, grow_up=0.5):
    """Wrist inside the tracked person's box, widened for outstretched / raised arms. No person -> not owned."""
    if box is None:
        return False
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return x1 - grow_x * bw <= pt[0] <= x2 + grow_x * bw and y1 - grow_up * bh <= pt[1] <= y2 + 0.1 * bh


class Debouncer:
    """Fires one event when a gesture is held (>= `need` of the last `window` gesture runs), with a cooldown."""

    def __init__(self, window=4, need=3, cooldown=1.0):
        self.window, self.need, self.cooldown = window, need, cooldown
        self.hist = {"Left": deque(maxlen=window), "Right": deque(maxlen=window)}
        self.active = {"Left": None, "Right": None}
        self.last_fire = {}

    def update(self, hands, now):
        seen = {h.side: h for h in hands if h.owned and h.side in self.hist}
        events = []
        for side, hist in self.hist.items():
            hand = seen.get(side)
            hist.append(hand.label if hand else None)
            label, n = Counter(hist).most_common(1)[0]
            stable = label if n >= self.need else None
            if stable == self.active[side]:
                continue
            self.active[side] = stable
            if stable and now - self.last_fire.get((side, stable), -1e9) >= self.cooldown:
                self.last_fire[(side, stable)] = now
                events.append(GestureEvent(now, side, stable, "custom" if hand and hand.custom == stable else "builtin"))
        return events
