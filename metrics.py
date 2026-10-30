"""Live stats for the on-screen panel. Keeps only the last few seconds in memory and saves nothing."""
import time
from collections import deque

import numpy as np


class Metrics:
    def __init__(self, window=150):
        self.recent = deque(maxlen=window)
        self.shown = deque()  # display timestamps for a 1-second rolling FPS

    def add_frame(self, rec):
        now = time.perf_counter()
        self.recent.append(rec)
        self.shown.append(now)
        while now - self.shown[0] > 1.0:
            self.shown.popleft()

    def live(self):
        r = self.recent
        avg = lambda key, keep=lambda x: True: float(np.mean([x[key] for x in r if keep(x)] or [0]))
        p95 = lambda key: float(np.percentile([x[key] for x in r], 95)) if r else 0.0
        return {
            "fps": len(self.shown),
            "proc_ms": avg("proc_ms"), "proc_p95": p95("proc_ms"),
            "lag_ms": avg("lag_ms"), "lag_p95": p95("lag_ms"),
            "pose_ms": avg("pose_ms", lambda x: x["state"] == "detect"),
            "vit_ms": avg("vit_ms", lambda x: x["state"] == "track"),
            "gesture_ms": avg("gesture_ms", lambda x: x["gesture_ms"] > 0),
            "det_pct": 100 * sum(x["state"] == "detect" for x in r) / max(1, len(r)),
        }
