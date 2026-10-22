# Assistive Face + Gesture Tracking

Real-time face tracking and hand-gesture recognition from a webcam, built as the vision system for an assistive robot for elderly and disabled people. The robot can **turn to face the person** and they can **give commands with simple hand gestures**, without touching anything.

Runs in real time on a MacBook Air M2 with no dedicated GPU. The models total **~7.8 M parameters (~16 MB)**.

## How it works

```
Webcam frame
   │
   ├─► YOLO26n-pose  (every 15th frame)  ──► finds the person + nose/eyes/ears ──► face box
   │        │
   │        └─► ViT tracker (all other frames) ──► follows the face cheaply
   │
   └─► MediaPipe Gesture Recognizer (every 2nd frame) ──► hands + gesture name
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
| **Performance monitor** | Live FPS/latency overlay plus a report table after every session |

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

**Live webcam testing** (~3,600 frames, 1280×720):

| Stat | Result |
|---|---|
| Sustainable speed | **~29 FPS** (real-time) |
| Delay, frame captured → on screen | **35 ms** average (56 ms p95) |
| Face found | **97%** of frames |
| Frames needing the heavy detector | **10%** (the ViT tracker handles the rest) |
| Time per step | YOLO 18 ms · ViT 5 ms · gestures 25 ms |
| Gesture confidence (on MediaPipe's sample gesture photos) | 0.73 – 0.93 |

Displayed FPS also depends on the webcam. MacBook cameras drop to about 19 FPS in dim light. The report shows the camera's wait separately, so you can see whether the camera or the code is the bottleneck.

**Why these models?** An earlier benchmark on the same machine compared YOLO26n/26s/11n/v8n with several trackers:
- **YOLO + ViT was the lightest setup:** 101.8 FPS, with the detector on only 12.7% of frames and 0 identity switches. YOLOv8n switched identity 6–7 times.
- **Head-pose test (13 poses: tilts, turns, looking up/down, back of head):** the pose model found the head in 100% of frames. The YuNet face detector dropped to 9–26% when the person looked up or down.

## Setup

Works on **macOS (Apple silicon)** and **Ubuntu 22.04 / 24.04** (x86_64 PCs and ARM boards), with Python 3.12 or 3.13.

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
- `libportaudio2`: MediaPipe loads its audio module at startup;
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
| `PortAudio library not found` | `sudo apt install libportaudio2` |
| `libEGL.so.1` / `libGLESv2.so.2` not found | `sudo apt install libegl1 libgles2` |
| `libGL.so.1` / `libgthread` errors | `sudo apt install libgl1 libglib2.0-0` |
| Window doesn't open (Wayland desktop) | `QT_QPA_PLATFORM=xcb .venv/bin/python app.py` |
| `could not connect to display` | You're over SSH without a display: reconnect with `ssh -X`, or run it on the machine's desktop |
| Low FPS on a CPU-only PC | `--gesture-every 3` and/or `--redetect 20` |
| NVIDIA Jetson runs on CPU | PyTorch from pip has no Jetson GPU support. Install NVIDIA's JetPack PyTorch build first. |

Every push is automatically tested on Ubuntu 22.04, Ubuntu 24.04 and Ubuntu 24.04 ARM through GitHub Actions (`.github/workflows/ubuntu.yml`).

On macOS, the first run asks for camera permission for your terminal or VS Code.

## Usage

```bash
.venv/bin/python app.py
```

| Key | Action |
|---|---|
| `q` / `Esc` | Quit (saves the session report) |
| `r` | Force the face to be re-detected |
| `d` | Show or hide the stats overlay |

Useful options:

```bash
.venv/bin/python app.py --redetect 10       # correct the tracker more often (less drift, a bit slower)
.venv/bin/python app.py --gesture-every 1   # check gestures every frame (more responsive, more load)
.venv/bin/python app.py --device cpu        # force CPU only (no GPU)
.venv/bin/python app.py --help              # all options
```

### Custom gestures

```bash
.venv/bin/python app.py --record help_me
```

Press `c` to start recording, hold the gesture and move your hand slightly, then press `c` to stop. About 100 samples takes around 7 seconds. The gesture works immediately and is saved in `custom_gestures/samples.csv` for future runs. Record one label per run.

A model trained with [MediaPipe Model Maker](https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer) (in Colab) can also be loaded with `--gesture-model my_gestures.task`.

### Reports

When you quit, each session is saved to `runs/<timestamp>/`:
- `summary.md`: a table of FPS, latency, face-found %, drift and gestures detected;
- `frames.csv` and `events.csv`: the raw data for every frame and every gesture.

To compare all sessions side by side:

```bash
.venv/bin/python report.py    # writes runs/all_runs.md
```

## Project structure

```
app.py            webcam loop, overlay, keys, CLI
face_tracker.py   YOLO26n-pose + ViT hybrid face tracker
gestures.py       MediaPipe gestures, custom k-NN gestures, owner filter, debouncer
metrics.py        live stats for the overlay
report.py         session report tables
smoke_test.py     headless self-check (used by the Ubuntu CI)
setup.sh          one-time environment + model download
requirements.txt  pinned package versions
```

For a robot, each frame provides the face position as an offset from the image centre (−1 to +1 on each axis), ready to drive a pan/tilt head, plus a list of gesture events.

## Limitations and next steps

- **Tracking only:** it follows *a* person but doesn't know *who*. Face recognition is the next step.
- **Drift:** the tracker's re-detect agreement (IoU) is about 0.65. Lower `--redetect` if the box drifts.
- **Robot integration:** port to the robot's onboard computer, connect the face offset to the head motors, and map gestures to actions.

## Credits and licenses

- [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics): AGPL-3.0
- [MediaPipe](https://github.com/google-ai-edge/mediapipe): Apache-2.0
- [OpenCV](https://github.com/opencv/opencv) and the [ViT tracker from OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack): Apache-2.0

Because YOLO is AGPL-3.0, publicly distributing this project, or offering it as a networked service, requires releasing the source under compatible terms.
