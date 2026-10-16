"""Live performance stats for the HUD + saving the session (frames.csv, events.csv, summary.md via report.py)."""
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
import psutil

from report import write_run_report

FRAME_FIELDS = ["frame", "t", "state", "read_ms", "pose_ms", "vit_ms", "gesture_ms", "draw_ms", "proc_ms",
                "lag_ms", "gesture_run", "hands", "redetect_iou"]


def pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


class Metrics:
    def __init__(self, window=300):
        self.t0 = time.perf_counter()
        self.frames, self.events = [], []
        self.recent = deque(maxlen=window)
        self.shown = deque()  # display timestamps for a 1-second rolling FPS
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        self.cpu, self.ram_mb, self._sys_t = 0.0, 0.0, 0.0

    def add_frame(self, rec):
        now = time.perf_counter()
        rec["t"] = round(now - self.t0, 3)
        self.frames.append(rec)
        self.recent.append(rec)
        self.shown.append(now)
        while self.shown and now - self.shown[0] > 1.0:
            self.shown.popleft()
        if now - self._sys_t > 0.5:
            self.cpu, self.ram_mb, self._sys_t = self.proc.cpu_percent(None), self.proc.memory_info().rss / 2**20, now

    def add_event(self, ev):
        self.events.append({"t": round(ev.t - self.t0, 3), "side": ev.side, "gesture": ev.label, "source": ev.source})

    def live(self):
        r = list(self.recent)
        col = lambda k: [x[k] for x in r if x[k] is not None]
        mean = lambda k: float(np.mean(col(k))) if col(k) else 0.0
        gest = [x["gesture_ms"] for x in r if x["gesture_run"]]
        ious = [x["redetect_iou"] for x in r if x["redetect_iou"] is not None]
        return {
            "fps": len(self.shown),
            "proc_ms": mean("proc_ms"), "proc_p95": pct(col("proc_ms"), 95),
            "lag_ms": mean("lag_ms"), "lag_p95": pct(col("lag_ms"), 95),
            "pose_ms": float(np.mean([x["pose_ms"] for x in r if x["state"] == "detect"] or [0])),
            "vit_ms": float(np.mean([x["vit_ms"] for x in r if x["state"] == "track"] or [0])),
            "gesture_ms": float(np.mean(gest)) if gest else 0.0,
            "draw_ms": mean("draw_ms"),
            "det_pct": 100 * sum(x["state"] == "detect" for x in r) / max(1, len(r)),
            "redetect_iou": float(np.mean(ious)) if ious else float("nan"),
            "cpu": self.cpu, "ram_mb": self.ram_mb,
        }

    def write_report(self, out_dir, config):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "frames.csv", "w", newline="") as f:
            w = csv.DictWriter(f, FRAME_FIELDS)
            w.writeheader()
            w.writerows(self.frames)
        with open(out / "events.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["t", "side", "gesture", "source"])
            w.writeheader()
            w.writerows(self.events)
        return write_run_report(out, self.frames, self.events, config)
