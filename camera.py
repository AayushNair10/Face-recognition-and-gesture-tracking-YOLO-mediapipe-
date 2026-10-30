"""Frame source: a webcam read on its own thread, or a video file for repeatable tests."""
import sys
import threading
import time

import cv2


def _open_webcam(index, width, height):
    backend = {"darwin": cv2.CAP_AVFOUNDATION, "linux": cv2.CAP_V4L2}.get(sys.platform, cv2.CAP_ANY)
    cap = cv2.VideoCapture(index, backend)
    if sys.platform == "linux":  # most USB webcams only reach 30 FPS at 720p in MJPG mode
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by some drivers; the capture thread drops old frames anyway
    for _ in range(50):  # the first reads can fail while the camera wakes up
        ok, _ = cap.read()
        if ok:
            return cap
        time.sleep(0.05)
    hint = {
        "darwin": "On macOS allow camera access for Terminal / VS Code in System Settings > Privacy & Security > Camera.",
        "linux": "On Ubuntu check the camera exists (ls /dev/video*), try another --camera index, and make sure your "
                 "user is in the 'video' group (sudo usermod -aG video $USER, then log out and in).",
    }.get(sys.platform, "Check the camera is connected and not used by another app.")
    raise SystemExit(f"Could not read from camera {index}. {hint}")


class FrameSource:
    """read() returns (frame, capture_time), or (None, None) when the source ends.

    Webcam: a background thread keeps only the newest frame, so a slow step can never leave the robot
    working on an old picture. Video file: every frame, in order, so test runs are repeatable.
    """

    def __init__(self, source, width=1280, height=720):
        self.is_file = isinstance(source, str)
        self.cap = cv2.VideoCapture(source) if self.is_file else _open_webcam(source, width, height)
        if self.is_file and not self.cap.isOpened():
            raise SystemExit(f"Could not open video file: {source}")
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._frame, self._time, self._seq, self._seen = None, None, 0, 0
        self._new = threading.Condition()
        self._running = True
        if not self.is_file:
            self._thread = threading.Thread(target=self._capture, daemon=True)
            self._thread.start()

    def _capture(self):
        failures = 0
        while self._running:
            ok, frame = self.cap.read()
            now = time.perf_counter()
            failures = 0 if ok else failures + 1
            if failures > 30:
                break
            if ok:
                with self._new:
                    self._frame, self._time = frame, now
                    self._seq += 1
                    self._new.notify_all()
        with self._new:
            self._running = False
            self._new.notify_all()

    def read(self):
        if self.is_file:
            ok, frame = self.cap.read()
            return (frame, time.perf_counter()) if ok else (None, None)
        with self._new:
            while self._seq == self._seen and self._running:
                self._new.wait(timeout=1.0)
            if self._seq == self._seen:
                return None, None
            self._seen = self._seq
            return self._frame, self._time

    def release(self):
        self._running = False
        if not self.is_file:
            self._thread.join(timeout=2.0)
        self.cap.release()
