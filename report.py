"""Session reports as simple tables.

  .venv/bin/python report.py     # rebuild summary.md for every run in runs/ + runs/all_runs.md (one column per run)

app.py calls write_run_report() automatically when you quit.
"""
import csv
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import psutil

RUNS = Path(__file__).parent / "runs"
NUM = ["t", "read_ms", "pose_ms", "vit_ms", "gesture_ms", "draw_ms", "proc_ms", "lag_ms", "gesture_run", "hands"]


def _mean(v):
    return float(np.mean(v)) if len(v) else float("nan")


def _div(a, b):
    return a / b if b else float("nan")


def _p95(v):
    return float(np.percentile(v, 95)) if len(v) else float("nan")


def run_stats(frames, events):
    """Numbers for one run. frames/events: lists of dicts (from the app or read back from the CSVs)."""
    F = [{**r, **{k: float(r[k]) for k in NUM}} for r in frames]
    F = F[10:] if len(F) > 30 else F  # skip warm-up frames
    if len(F) < 2:
        return None
    col = lambda k, cond=lambda r: True: [r[k] for r in F if cond(r)]
    dur = F[-1]["t"] - F[0]["t"]
    gaps = np.diff([r["t"] for r in F]) * 1000
    lost = sum(r["state"] == "lost" for r in F)
    ious = [float(r["redetect_iou"]) for r in F if r.get("redetect_iou") not in (None, "")]
    return {
        "duration_s": dur, "frames": len(F),
        "fps": _div(len(F), dur), "fps_worst": _div(1000, _p95(gaps)),
        "camera_wait_ms": _mean(col("read_ms")),
        "capacity_fps": _div(1000, _mean(col("lag_ms"))),  # what the pipeline could do if the camera never made it wait
        "proc_ms": _mean(col("proc_ms")), "proc_p95": _p95(col("proc_ms")),
        "lag_ms": _mean(col("lag_ms")), "lag_p95": _p95(col("lag_ms")),
        "pose_ms": _mean(col("pose_ms", lambda r: r["state"] == "detect")),
        "vit_ms": _mean(col("vit_ms", lambda r: r["state"] == "track")),
        "gesture_ms": _mean(col("gesture_ms", lambda r: r["gesture_run"])),
        "draw_ms": _mean(col("draw_ms")),
        "face_found_pct": 100 * (1 - lost / len(F)),
        "face_lost_times": sum(1 for a, b in zip(F, F[1:]) if a["state"] != "lost" and b["state"] == "lost"),
        "detector_pct": 100 * sum(r["state"] == "detect" for r in F) / len(F),
        "drift_iou": _mean(ious),
        "gesture_runs_pct": 100 * sum(r["gesture_run"] for r in F) / len(F),
        "events": len(events),
        "event_counts": Counter(e["gesture"] for e in events),
    }


# (label, value formatter, "good if" text, check -> True/False/None)
ROWS = [
    ("Duration", lambda s: f"{s['duration_s']:.0f} s ({s['frames']} frames)", "", None),
    ("Average FPS", lambda s: f"{s['fps']:.1f}", ">= 25", lambda s: s["fps"] >= 25),
    ("Worst-case FPS (slowest 5%)", lambda s: f"{s['fps_worst']:.1f}", ">= 15", lambda s: s["fps_worst"] >= 15),
    ("Camera wait per frame", lambda s: f"{s['camera_wait_ms']:.1f} ms", "< 5 ms (else the camera is the bottleneck)",
     lambda s: s["camera_wait_ms"] < 5),
    ("FPS the code could reach (no camera wait)", lambda s: f"{s['capacity_fps']:.1f}", ">= 25", lambda s: s["capacity_fps"] >= 25),
    ("Processing per frame (avg / p95)", lambda s: f"{s['proc_ms']:.1f} / {s['proc_p95']:.1f} ms", "avg < 33 ms",
     lambda s: s["proc_ms"] < 33),
    ("Lag: frame read -> on screen (avg / p95)", lambda s: f"{s['lag_ms']:.1f} / {s['lag_p95']:.1f} ms", "avg < 100 ms",
     lambda s: s["lag_ms"] < 100),
    ("Face found", lambda s: f"{s['face_found_pct']:.1f}%", ">= 95%", lambda s: s["face_found_pct"] >= 95),
    ("Times face was lost", lambda s: f"{s['face_lost_times']}", "", None),
    ("Frames using the YOLO detector", lambda s: f"{s['detector_pct']:.1f}%", "low (rest is the cheap ViT tracker)", None),
    ("Tracker drift check (re-detect IoU)", lambda s: f"{s['drift_iou']:.2f}", ">= 0.6 (1.0 = no drift)",
     lambda s: s["drift_iou"] >= 0.6),
    ("YOLO pose time (when it runs)", lambda s: f"{s['pose_ms']:.1f} ms", "", None),
    ("ViT tracker time", lambda s: f"{s['vit_ms']:.1f} ms", "", None),
    ("Gesture model time (when it runs)", lambda s: f"{s['gesture_ms']:.1f} ms", "", None),
    ("Gesture model ran on", lambda s: f"{s['gesture_runs_pct']:.0f}% of frames", "", None),
    ("Gestures detected", lambda s: f"{s['events']}" + (" (" + ", ".join(f"{k} x{v}" for k, v in s["event_counts"].most_common()) + ")"
                                                      if s["events"] else ""), "", None),
]


def _mark(check, s):
    if check is None:
        return ""
    try:
        return "OK" if check(s) else "LOW"
    except (TypeError, ValueError):
        return ""


def run_table(name, s, config):
    setup = ", ".join(f"{k}={v}" for k, v in config.items() if k != "custom_labels")
    L = [f"# Session report: {name}", "",
         f"Setup: {platform.machine()} Mac, {psutil.virtual_memory().total / 2**30:.0f} GB RAM, {setup}", "",
         "| Metric | Result | Good if | Status |", "|---|---|---|---|"]
    L += [f"| {lbl} | {fmt(s)} | {good} | {_mark(chk, s)} |" for lbl, fmt, good, chk in ROWS]
    L += ["", "Lag does not include the webcam's own delay (not measurable in software, typically 30-60 ms).", ""]
    return "\n".join(L)


# short plain-language rows for the all-runs comparison
SIMPLE = [
    ("Run length", lambda s: f"{s['duration_s']:.0f} s", ""),
    ("Speed (FPS)", lambda s: f"{s['fps']:.1f}", "25+"),
    ("Speed the code can do (FPS)", lambda s: f"{s['capacity_fps']:.1f}", "25+"),
    ("Delay on screen", lambda s: f"{s['lag_ms']:.0f} ms", "under 100 ms"),
    ("Face found", lambda s: f"{s['face_found_pct']:.0f}%", "95%+"),
    ("Face lost (times)", lambda s: f"{s['face_lost_times']}", "0"),
    ("Gestures caught", lambda s: f"{s['events']}", ""),
]


def compare_table(runs):
    """runs: list of (name, stats). One row per metric, one column per run."""
    names = [f"Run {i} ({n[9:11]}:{n[11:13]})" for i, (n, _) in enumerate(runs, 1)]
    L = ["# All runs", "", "| | " + " | ".join(names) + " | Goal |", "|---|" + "---|" * (len(runs) + 1)]
    for lbl, fmt, goal in SIMPLE:
        L.append(f"| {lbl} | " + " | ".join(fmt(s) for _, s in runs) + f" | {goal} |")
    L += ["", "Full details per run: runs/<run>/summary.md"]
    return "\n".join(L) + "\n"


def write_run_report(run_dir, frames, events, config):
    s = run_stats(frames, events)
    md = run_table(Path(run_dir).name, s, config) if s else "No frames recorded.\n"
    (Path(run_dir) / "summary.md").write_text(md)
    (Path(run_dir) / "config.json").write_text(json.dumps(config, indent=1))
    return md


def _load_config(d):
    if (d / "config.json").exists():
        return json.loads((d / "config.json").read_text())
    for line in (d / "summary.md").read_text().splitlines() if (d / "summary.md").exists() else []:
        if line.startswith("Config: "):  # reports written before config.json existed
            return dict(kv.split("=", 1) for kv in line[8:].split(", ") if "=" in kv)
    return {}


def main():
    runs = []
    for d in sorted(p for p in RUNS.glob("*") if (p / "frames.csv").exists()):
        frames = list(csv.DictReader(open(d / "frames.csv")))
        events = list(csv.DictReader(open(d / "events.csv"))) if (d / "events.csv").exists() else []
        s = run_stats(frames, events)
        if s is None:
            continue
        write_run_report(d, frames, events, _load_config(d))
        runs.append((d.name, s))
    if not runs:
        raise SystemExit("no runs found in runs/")
    md = compare_table(runs)
    (RUNS / "all_runs.md").write_text(md)
    print(md)
    print(f"written: runs/all_runs.md + summary.md in each of {len(runs)} run folder(s)")


if __name__ == "__main__":
    main()
