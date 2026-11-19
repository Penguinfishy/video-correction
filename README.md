# Band Stabilizer (macOS, Apple Silicon)

Local, gentle video stabilization for steady wide shots. It measures how the phone
rotated (yaw, pitch, roll) from the bleachers/stands you select, then re-renders each
frame as if the phone had been held at a smoothed orientation (lens-correct 3D
reprojection, one fixed slight zoom). Held shots lock to one average orientation;
deliberate pans are kept. Moving performers are rejected as outliers.

In the app: Choose Video → Set Bleachers Area (drag a box over the stands) → Stabilize.

## Setup (once)
```bash
brew install python@3.13 python-tk@3.13 ffmpeg
cd ~/video-correction
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/pip install opencv-python-headless numpy
```

## Run (returns Terminal to you immediately)
```bash
open ~/Desktop/"Band Stabilizer.app"
```
Before the launcher exists: `(.venv/bin/python app.py >/dev/null 2>&1 &)`
Double-click launcher: `./make_app.sh` creates `Band Stabilizer.app` on the Desktop
(re-run it if you move this folder).

Output: `yourvideo_stabilized.mp4` next to the original (H.264 + original audio).
Command line alternative: `.venv/bin/python stabilize_core.py video.mov Gentle`

Motion analysis is saved as `yourvideo_motion.npz` next to the original. Re-running
with a different strength reuses it and only re-renders. Delete it to force a fresh analysis.
