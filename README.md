# Band Stabilizer (macOS, Apple Silicon)

Local, gentle video stabilization for steady wide shots. ffmpeg decodes/encodes;
OpenCV tracks stationary background points (RANSAC rejects moving performers),
corrects only rotation + small shifts with one fixed slight zoom, holds a steady
average framing during held shots, and follows deliberate pans/zooms.

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
