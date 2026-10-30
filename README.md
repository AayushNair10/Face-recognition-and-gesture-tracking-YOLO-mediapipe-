# Assistive Face + Gesture Tracking

Real-time face tracking and hand-gesture recognition from a webcam, built as the vision system for an assistive robot for elderly and disabled people. The robot can **turn to face the person** and they can **give commands with simple hand gestures**, without touching anything.

Runs in real time on a MacBook Air M2 with no dedicated GPU, and fully offline. The models total **~7.8 M parameters (~16 MB)**.

## How it works

```
Webcam frame
   │
   ├─► YOLO26n-pose  (every 15th frame)  ──► finds the person + nose/eyes/ears ──► face box
   │        │
   │        └─► ViT tracker (all other frames) ──► follows the face cheaply
   │
   └─► MediaPipe Gesture Recognizer (own thread, every 2nd frame) ──► hands + gesture name
            │
            ├─► owner filter: only the tracked person's hands count
            └─► hold check: gesture must be held ~0.3 s → fires ONE event
```

| Component | Role |
|---|---|
| **YOLO26n-pose** | Detects the person and the face keypoints; the face box is built from them |
| **OpenCV ViT tracker** | Follows the face between detections at a fraction of the cost |
| **MediaPipe Gesture Recognizer** | Finds up to 2 hands (21 landmarks each) and classifies the gesture |
| **Owner filter** | Ignores hands that don't belong to the tracked person |
| **Debouncer** | Stops flicker and repeated triggers |
| **Custom gestures** | Teach new gestures in seconds (k-NN on hand landmarks, no retraining) |
| **Stats panel** | Live FPS and latency on screen; nothing is saved |

**Built-in gestures:** Open palm, Closed fist, Thumbs up, Thumbs down, Pointing up, Victory, I-love-you.

## Models and parameters

| Model | Job | Parameters |
|---|---|---|
| YOLO26n-pose | Person + face keypoints | 3.68 M |
| ViT tracker (OpenCV Zoo, 2023-sep) | Face following | 0.17 M |
| MediaPipe Gesture Recognizer | Hands + gestures | 3.98 M |
| ↳ hand detector / hand landmarks / gesture embedder / classifier | | 1.13 M / 2.72 M / 0.13 M / 1.3 K |
| **Total** | | **≈ 7.8 M** |

## Performance (MacBook Air M2, 8 GB)

**Live webcam test** (25 s, 745 frames, 1280×720):

| Stat | Result |
|---|---|
| Speed | **30 FPS** (the webcam's limit; the code can reach ~50 FPS) |
| Delay, frame captured → on screen | **20 ms** average (30 ms p95) |
| Face found | **100%** of frames |
| Frames needing the heavy detector | **6%** (the ViT tracker handles the rest) |
| Time per step | YOLO 15 ms · ViT 4 ms · gestures 19 ms |
| Gesture confidence (on MediaPipe's sample gesture photos) | 0.73 – 0.93 |

Speed depends on the webcam: MacBook cameras can drop to about 19 FPS in dim light.

**Why these models?** An earlier benchmark on the same machine compared YOLO26n/26s/11n/v8n with several trackers:
- **YOLO + ViT was the lightest setup:** 101.8 FPS, with the detector on only 12.7% of frames and 0 identity switches. YOLOv8n switched identity 6–7 times.
- **Head-pose test (13 poses: tilts, turns, looking up/down, back of head):** the pose model found the head in 100% of frames. The YuNet face detector dropped to 9–26% when the person looked up or down.

## Setup

Works on **macOS (Apple silicon)**, **Ubuntu 22.04 / 24.04** (x86_64 PCs and ARM boards) and **NVIDIA Jetson** (JetPack 5, 6 and 7), with Python 3.12 or 3.13 (Jetson uses its system Python).

Setup needs internet once. After that the app runs completely offline.

```bash
git clone https://github.com/AayushNair10/Face-recognition-and-gesture-tracking-YOLO-mediapipe-.git
cd Face-recognition-and-gesture-tracking-YOLO-mediapipe-
bash setup.sh                    # venv, packages, the 3 model files (+ system packages on Ubuntu)
.venv/bin/python smoke_test.py   # checks everything works, no webcam needed
```

To use a particular Python, run `PY=python3.12 bash setup.sh`.

The models run on the best hardware available: an NVIDIA GPU (`cuda`), then the Apple GPU (`mps`), then the CPU.

### Running on an Ubuntu machine

**1. Install the basics.** Ubuntu 24.04 already has Python 3.12:
```bash
sudo apt update && sudo apt install -y git curl python3 python3-venv
```
Ubuntu 22.04 ships Python 3.10, so install 3.12 alongside it:
```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.12 python3.12-venv
```

**2. Get the code and set it up:**
```bash
git clone https://github.com/AayushNair10/Face-recognition-and-gesture-tracking-YOLO-mediapipe-.git
cd Face-recognition-and-gesture-tracking-YOLO-mediapipe-
bash setup.sh                    # on 22.04: PY=python3.12 bash setup.sh
```
`setup.sh` asks for your sudo password once to install the system packages it needs:
- `libegl1` and `libgles2`: MediaPipe's native library needs OpenGL;
- `libgl1` and `libglib2.0-0`: needed for the OpenCV window.

On a PC without an NVIDIA GPU it installs the CPU build of PyTorch (~200 MB instead of ~3 GB).

**3. Check that everything works (no webcam needed):**
```bash
.venv/bin/python smoke_test.py   # should end with: ALL PASSED
```

**4. Check the camera, then run the app:**
```bash
ls /dev/video*                   # should list at least /dev/video0
.venv/bin/python app.py          # use --camera 1, 2, ... if video0 isn't your webcam
```
Run it from the Ubuntu desktop, because it opens a window. Over SSH, use `ssh -X` so the window shows on your own machine.

**Troubleshooting**

| Problem | Fix |
|---|---|
| `Could not read from camera` | Try another index (`--camera 1`). Add yourself to the video group with `sudo usermod -aG video $USER`, then log out and back in. Close other apps that are using the camera. |
| `libEGL.so.1` / `libGLESv2.so.2` not found | `sudo apt install libegl1 libgles2` |
| `libGL.so.1` / `libgthread` errors | `sudo apt install libgl1 libglib2.0-0` |
| Window doesn't open (Wayland desktop) | `QT_QPA_PLATFORM=xcb .venv/bin/python app.py` |
| `could not connect to display` | You're over SSH without a display: reconnect with `ssh -X`, or run it on the machine's desktop |
| Low FPS on a CPU-only PC | `--gesture-every 3` and/or `--redetect 20` |
| Jetson runs on CPU | Re-run `setup.sh` on the Jetson: it installs the PyTorch build for your JetPack |

Every push is automatically tested with no network on Ubuntu 22.04, Ubuntu 24.04 and Ubuntu 24.04 ARM through GitHub Actions (`.github/workflows/ubuntu.yml`).

### Running on the robot (NVIDIA Jetson, offline)

Connect the Jetson to the internet once (Ethernet or a phone hotspot), run `bash setup.sh`, then disconnect. `setup.sh` detects your JetPack version and installs the matching GPU build of PyTorch. Then:

```bash
.venv/bin/python smoke_test.py      # should end with ALL PASSED and show device cuda
.venv/bin/python app.py --headless  # no window, nothing saved
```

Robot code goes in `on_result()` in `app.py`, which runs once per frame with the face position and any gestures that just fired.

For maximum speed, set the Jetson to full power once per boot: `sudo nvpmodel -m 0 && sudo jetson_clocks`.

| JetPack | Python | Jetson models | Status |
|---|---|---|---|
| 7.2 | 3.12 | Orin family | Supported |
| 6.x | 3.10 | Orin family | Supported |
| 5.x | 3.8 | Xavier, Orin | Supported |
| 4.x | 3.6 | Original Jetson Nano, TX2 | Not supported (too old for MediaPipe and Ultralytics) |

**Camera:** use a USB camera. The OpenCV installed here has no GStreamer, so a ribbon-cable (CSI) camera needs extra work.

### Required packages

Everything installs into `.venv`, so nothing touches the robot's system Python.

| Package | Version | Used for |
|---|---|---|
| `ultralytics` | 8.4.174 | Runs YOLO26n-pose |
| `torch`, `torchvision` | 2.14.1 / 0.29.1 (Jetson: the JetPack build) | Runs YOLO on the GPU |
| `mediapipe` | 1.1.1 | Hand gestures |
| `opencv-contrib-python` | 5.0.0.93 | Camera, ViT tracker, window |
| `numpy` | newest for your Python | Arrays |

These pull in about 40 smaller packages, including `matplotlib`, `polars`, `pillow`, `pyyaml`, `requests` and `sounddevice`. System packages (Ubuntu and Jetson): `libegl1`, `libgles2`, plus `libgl1`, `libglib2.0-0`, `libxcb-xinerama0` and `libxkbcommon-x11-0` for the window.

To avoid conflicts with other robot software, especially ROS:
- keep this project in its own `.venv`, and don't install its packages system-wide;
- run it as its own process, because OpenCV and NumPy here can differ from the versions ROS uses;
- never have both `opencv-python` and `opencv-contrib-python` in one environment. `setup.sh` removes the first.

On macOS, the first run asks for camera permission for your terminal or VS Code.

## Usage

```bash
.venv/bin/python app.py
```

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `r` | Force the face to be re-detected |
| `d` | Show or hide the stats overlay |

Useful options:

```bash
.venv/bin/python app.py --headless          # robot mode: no window, nothing saved
.venv/bin/python app.py --imgsz 320         # faster YOLO, shorter range (default 416, reliable to ~4 m)
.venv/bin/python app.py --redetect 10       # correct the tracker more often (less drift, a bit slower)
.venv/bin/python app.py --gesture-every 1   # check gestures every frame (more responsive, more load)
.venv/bin/python app.py --source clip.mp4   # use a video file instead of the camera
.venv/bin/python app.py --device cpu        # force CPU only (no GPU)
.venv/bin/python app.py --help              # all options
```

### Custom gestures

```bash
.venv/bin/python app.py --record help_me
```

Press `c` to start recording, hold the gesture and move your hand slightly, then press `c` to stop. About 100 samples takes around 7 seconds. The gesture works immediately and is saved in `custom_gestures/samples.csv` for future runs. Record one label per run.

A model trained with [MediaPipe Model Maker](https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer) (in Colab) can also be loaded with `--gesture-model my_gestures.task`.

## Project structure

```
app.py            main loop, overlay, keys, CLI; on_result() is the robot hook
camera.py         newest-frame camera reader on its own thread
face_tracker.py   YOLO26n-pose + ViT hybrid face tracker
gestures.py       MediaPipe gestures on their own thread, custom k-NN gestures, owner filter, debouncer
metrics.py        live stats for the overlay
smoke_test.py     headless self-check (used by the Ubuntu CI)
setup.sh          one-time environment + model download
requirements.txt  pinned package versions
requirements-jetson.txt  the same, minus PyTorch (setup.sh installs the JetPack build)
```

For a robot, each frame provides the face position as an offset from the image centre (−1 to +1 on each axis), ready to drive a pan/tilt head, plus a list of gesture events.

## Limitations and next steps

- **Tracking only:** it follows *a* person but doesn't know *who*. Face recognition is the next step.
- **Drift:** the tracker's re-detect agreement (IoU) is about 0.65. Lower `--redetect` if the box drifts.
- **Robot integration:** connect the face offset in `on_result()` to the head motors, and map gestures to actions.
- **Jetson:** setup is automated but hasn't yet been run on a real Jetson.

## Credits and licenses

- [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics): AGPL-3.0
- [MediaPipe](https://github.com/google-ai-edge/mediapipe): Apache-2.0
- [OpenCV](https://github.com/opencv/opencv) and the [ViT tracker from OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack): Apache-2.0

Because YOLO is AGPL-3.0, publicly distributing this project, or offering it as a networked service, requires releasing the source under compatible terms.
