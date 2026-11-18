"""Stabilization engine: ffmpeg decodes/encodes, OpenCV estimates camera motion.

Pass 1: decode small grayscale frames, track background corners with optical
        flow, fit a rotation+translation(+uniform scale) model per frame pair
        with RANSAC (moving performers fall out as outliers).
Pass 2: decode full-resolution frames, apply one smooth correction per frame
        with a fixed slight zoom, encode, and mux the untouched source audio.
"""
import json
import math
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

# zoom = fixed crop for the whole video; sigma = seconds of smoothing during camera moves
STRENGTHS = {
    "Gentle": dict(zoom=1.04, sigma=1.0),
    "Medium": dict(zoom=1.06, sigma=1.6),
    "Strong": dict(zoom=1.08, sigma=2.5),
}
ANALYSIS_WIDTH = 640
COPYABLE_AUDIO = {"aac"}
DECODABLE_AUDIO_HINT = {"apac"}  # Apple spatial audio: ffmpeg can't read it


class StabilizeError(Exception):
    pass


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- tools/probe
def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    for d in ("/opt/homebrew/bin", "/usr/local/bin"):
        c = os.path.join(d, name)
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _frac(s):
    try:
        n, d = s.split("/")
        n, d = int(n), int(d)
        return (n, d) if n > 0 and d > 0 else None
    except Exception:
        return None


def probe(ffprobe, path):
    r = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format",
                        "-of", "json", path], capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise StabilizeError(f"This file can't be read as a video.\n\n{msg}")
    info = json.loads(r.stdout or "{}")
    streams = info.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic")]
    if not vids:
        raise StabilizeError("No video track found in this file.")
    v = vids[0]
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    if w < 64 or h < 64:
        raise StabilizeError("Unsupported video (missing or tiny frame size).")
    rot = 0
    try:
        rot = int(float(v.get("tags", {}).get("rotate", 0)))
    except ValueError:
        pass
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if abs(rot) % 180 == 90:
        w, h = h, w
    fr = _frac(v.get("avg_frame_rate", "")) or _frac(v.get("r_frame_rate", ""))
    if not fr or not (1 <= fr[0] / fr[1] <= 240):
        raise StabilizeError("Couldn't determine the video frame rate.")
    fmt = info.get("format", {})
    dur = float(fmt.get("duration") or v.get("duration") or 0)
    if dur <= 0:
        raise StabilizeError("Couldn't determine the video duration.")
    auds = [s for s in streams if s.get("codec_type") == "audio"]

    def start(s):
        try:
            return float(s.get("start_time", 0))
        except (TypeError, ValueError):
            return 0.0

    starts = [start(v)] + [start(a) for a in auds]
    bitrate = int(v.get("bit_rate") or 0) or int(fmt.get("bit_rate") or 0)
    return dict(w=w, h=h, fps_str=f"{fr[0]}/{fr[1]}", fps=fr[0] / fr[1], duration=dur,
                audio=auds, video_offset=max(0.0, start(v) - min(starts)), bitrate=bitrate)


def pick_encoder(ffmpeg):
    """Apple hardware H.264 if it actually works, else libx264."""
    test = [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=1280x720:d=0.2",
            "-pix_fmt", "yuv420p", "-f", "null", "-"]
    for enc in ("h264_videotoolbox", "libx264"):
        r = subprocess.run(test[:-3] + ["-c:v", enc] + test[-3:], capture_output=True)
        if r.returncode == 0:
            return enc
    raise StabilizeError("ffmpeg has no usable H.264 encoder. Reinstall with: brew reinstall ffmpeg")


# ---------------------------------------------------------------- decoding
class Decoder:
    def __init__(self, ffmpeg, path, fps_str, w, h, pix_fmt, vf=None):
        self.err = tempfile.TemporaryFile()
        cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0",
               "-an", "-sn", "-dn"]
        if vf:
            cmd += ["-vf", vf]
        # constant frame rate at the source's average rate keeps duration and A/V sync
        cmd += ["-fps_mode", "cfr", "-r", fps_str, "-pix_fmt", pix_fmt, "-f", "rawvideo", "-"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self.err,
                                  stdin=subprocess.DEVNULL, bufsize=1 << 20)
        self.shape = (h, w) if pix_fmt == "gray" else (h, w, 3)
        self.n = int(np.prod(self.shape))

    def read(self):
        buf = self.p.stdout.read(self.n)
        if len(buf) < self.n:
            return None
        return np.frombuffer(buf, np.uint8).reshape(self.shape)

    def close(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()
        self.err.seek(0)
        return self.err.read().decode(errors="replace").strip()


# ---------------------------------------------------------------- similarity math
def to_params(m):
    return np.array([math.atan2(m[1, 0], m[0, 0]), math.log(math.hypot(m[0, 0], m[1, 0])),
                     m[0, 2], m[1, 2]])


def to_mat(p):
    a, ls, tx, ty = p
    s = math.exp(ls)
    c, sn = s * math.cos(a), s * math.sin(a)
    return np.array([[c, -sn, tx], [sn, c, ty], [0.0, 0.0, 1.0]])


def gsmooth(x, sigma):
    x = np.asarray(x, float)
    if sigma <= 0.5 or len(x) < 2:
        return x.copy()
    r = int(3 * sigma) + 1
    t = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    if x.ndim == 1:
        return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")
    return np.stack([gsmooth(x[:, i], sigma) for i in range(x.shape[1])], axis=1)


def runs(mask):
    """[(start, end_exclusive, value)] for a boolean array."""
    out, s = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[s]:
            out.append((s, i, bool(mask[s])))
            s = i
    return out


# ---------------------------------------------------------------- pass 1
def estimate_motion(prev, cur, center):
    """Similarity transform (centered coords) mapping prev -> cur, from background points."""
    eye = np.eye(3)
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=600, qualityLevel=0.01,
                                  minDistance=10, blockSize=7)
    if pts is None or len(pts) < 15:
        return eye
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **lk)
    fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.7)
    if ok.sum() < 15:
        return eye
    p0 = pts[ok].reshape(-1, 2) - center
    p1 = nxt[ok].reshape(-1, 2) - center
    # RANSAC: the largest consistent group (stands, buildings, field markings) wins;
    # marching performers move inconsistently and are rejected as outliers.
    m, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=0.6,
                                         maxIters=3000, confidence=0.995, refineIters=20)
    if m is None or inl is None or inl.sum() < 12:
        return eye
    m3 = np.vstack([m, [0, 0, 1]])
    a, ls, _, _ = to_params(m3)
    if abs(a) > math.radians(5) or abs(ls) > 0.1:
        return eye
    return m3


def analyze(ffmpeg, src, info, aw, ah, progress, cancel):
    dec = Decoder(ffmpeg, src, info["fps_str"], aw, ah, "gray", vf=f"scale={aw}:{ah}:flags=area")
    center = np.array([aw / 2.0, ah / 2.0], np.float32)
    total = max(1, int(round(info["duration"] * info["fps"])))
    path, acc, prev = [], np.eye(3), None
    try:
        while True:
            if cancel.is_set():
                raise Cancelled()
            g = dec.read()
            if g is None:
                break
            if prev is not None:
                acc = estimate_motion(prev, g, center) @ acc  # frame0 -> frame t
            path.append(to_params(acc))
            prev = g
            if len(path) % 15 == 0:
                progress(0.4 * min(1.0, len(path) / total),
                         f"Analyzing motion… frame {len(path)} of ~{total}")
    finally:
        err = dec.close()
    if len(path) < 2:
        raise StabilizeError("Couldn't decode video frames.\n\n" + err)
    p = np.array(path)
    p[:, 0] = np.unwrap(p[:, 0])
    return p


# ---------------------------------------------------------------- planning
def plan(path, fps, aw, ah, zoom, sigma_s):
    """Return per-frame 3x3 correction matrices in centered analysis coordinates."""
    n = len(path)
    Z = np.diag([zoom, zoom, 1.0])
    hw, hh = aw / 2.0, ah / 2.0
    corners = np.array([[-hw, -hh, 1], [hw, -hh, 1], [hw, hh, 1], [-hw, hh, 1]], float).T
    lim = np.array([[hw * 0.999], [hh * 0.999]])
    A_inv = [np.linalg.inv(to_mat(p)) for p in path]

    def corr_params(target, i):
        return to_params(to_mat(target) @ A_inv[i])

    def fits(wp):
        src = np.linalg.inv(Z @ to_mat(wp)) @ corners
        return bool(np.all(np.abs(src[:2]) <= lim))

    # 1) classify intentional camera moves (pan/zoom/reframe) vs. held shots
    vel = np.gradient(gsmooth(path, 0.5 * fps), axis=0) * fps
    moving = ((np.hypot(vel[:, 2], vel[:, 3]) > 0.03 * aw)
              | (np.abs(vel[:, 0]) > math.radians(0.8))
              | (np.abs(vel[:, 1]) > 0.02))
    for s, e, v in runs(moving):  # ignore tiny blips
        if v and e - s < 0.3 * fps:
            moving[s:e] = False
    for s, e, v in runs(moving):  # very short holds just follow the move
        if not v and e - s < 1.5 * fps:
            moving[s:e] = True

    # 2) target camera path
    target = gsmooth(path, sigma_s * fps)
    for s, e, v in runs(moving):
        if v:
            continue
        seg = path[s:e]
        t = np.arange(e - s)
        cands = [np.repeat(seg.mean(0, keepdims=True), e - s, axis=0)]   # fixed average framing
        coef = np.polyfit(t, seg, 1)                                     # slow steady drift
        cands.append(np.outer(t, coef[0]) + coef[1])
        cands.append(gsmooth(seg, 3.0 * fps))                            # very slow wander
        for c in cands:
            if all(fits(corr_params(c[j], s + j)) for j in range(0, e - s, 2)):
                target[s:e] = c
                break
        else:
            target[s:e] = cands[-1]
    target = gsmooth(target, 0.3 * fps)  # soften transitions between holds and moves

    # 3) never show borders: shrink correction where it exceeds the fixed crop, smoothly
    wps = [corr_params(target[i], i) for i in range(n)]
    k_req = np.ones(n)
    for i, wp in enumerate(wps):
        if not fits(wp):
            lo, hi = 0.0, 1.0
            for _ in range(14):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if fits(wp * mid) else (lo, mid)
            k_req[i] = lo
    r = max(1, int(0.5 * fps))
    padded = np.pad(k_req, r, mode="edge")
    k_min = np.min(np.stack([padded[j:j + n] for j in range(2 * r + 1)]), axis=0)
    k = np.minimum(gsmooth(k_min, 0.25 * fps), k_req)
    return [Z @ to_mat(wps[i] * k[i]) for i in range(n)], float((k < 0.999).mean())


# ---------------------------------------------------------------- pass 2
def render(ffmpeg, src, tmp_out, info, W, H, mats, scale, encoder, progress, cancel):
    crop = f"crop={W}:{H}:0:0" if (W, H) != (info["w"], info["h"]) else None
    dec = Decoder(ffmpeg, src, info["fps_str"], W, H, "bgr24", vf=crop)
    usable = [a for a in info["audio"] if a.get("codec_name") not in (None, "none", "unknown")
              and a.get("codec_name") not in DECODABLE_AUDIO_HINT]
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-framerate", info["fps_str"]]
    if info["video_offset"] > 0.001:
        cmd += ["-itsoffset", f"{info['video_offset']:.6f}"]
    cmd += ["-i", "-", "-i", src, "-map", "0:v:0"]
    for a in usable:
        cmd += ["-map", f"1:{a['index']}"]
    for j, a in enumerate(usable):
        if a.get("codec_name") in COPYABLE_AUDIO:
            cmd += [f"-c:a:{j}", "copy"]
        else:
            cmd += [f"-c:a:{j}", "aac", f"-b:a:{j}", "256k"]
    cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    if encoder == "h264_videotoolbox":
        br = int(min(120e6, max(info["bitrate"] or 0, W * H * info["fps"] * 0.12)))
        cmd += ["-c:v", "h264_videotoolbox", "-b:v", str(br), "-profile:v", "high"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high"]
    cmd += ["-movflags", "+faststart", "-map_metadata", "1", "-f", "mp4", tmp_out]
    enc_err = tempfile.TemporaryFile()
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=enc_err)
    cf = np.array([W / 2.0, H / 2.0])
    total = len(mats)
    i = 0
    try:
        while True:
            if cancel.is_set():
                raise Cancelled()
            f = dec.read()
            if f is None:
                break
            m = mats[min(i, total - 1)].copy()
            m[:2, 2] *= scale
            lin = m[:2, :2]
            M = np.hstack([lin, (m[:2, 2] + cf - lin @ cf)[:, None]])
            out = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            try:
                enc.stdin.write(out.tobytes())
            except BrokenPipeError:
                break
            i += 1
            if i % 10 == 0:
                progress(0.4 + 0.6 * min(1.0, i / total), f"Rendering… frame {i} of {total}")
        enc.stdin.close()
        rc = enc.wait()
    finally:
        if enc.poll() is None:
            enc.kill()
            enc.wait()
        dec_err = dec.close()
    enc_err.seek(0)
    msg = enc_err.read().decode(errors="replace").strip()
    if rc != 0 or i == 0:
        raise StabilizeError(f"Encoding failed ({encoder}).\n\n{msg or dec_err}"[-1500:])
    return len(usable), len(info["audio"]) - len(usable)


# ---------------------------------------------------------------- main entry
def output_path(src):
    return os.path.splitext(src)[0] + "_stabilized.mp4"


def check_dependencies():
    ff, fp = find_tool("ffmpeg"), find_tool("ffprobe")
    if not ff or not fp:
        raise StabilizeError("ffmpeg was not found.\n\nInstall it in Terminal with:\n"
                             "brew install ffmpeg")
    return ff, fp


def stabilize(src, dst, strength="Gentle", progress=lambda f, m: None, cancel=None):
    import threading
    cancel = cancel or threading.Event()
    ffmpeg, ffprobe = check_dependencies()
    info = probe(ffprobe, src)
    cfg = STRENGTHS[strength]
    W, H = info["w"] - info["w"] % 2, info["h"] - info["h"] % 2
    aw = min(ANALYSIS_WIDTH, W - W % 2)
    ah = max(2, int(round(H * aw / W / 2)) * 2)

    progress(0.0, "Checking encoder…")
    encoder = pick_encoder(ffmpeg)
    path = analyze(ffmpeg, src, info, aw, ah, progress, cancel)
    progress(0.4, "Planning smooth camera path…")
    mats, limited = plan(path, info["fps"], aw, ah, cfg["zoom"], cfg["sigma"])

    d, b = os.path.split(dst)
    tmp = os.path.join(d, "." + b + ".partial.mp4")
    try:
        try:
            kept, skipped = render(ffmpeg, src, tmp, info, W, H, mats, W / aw, encoder,
                                   progress, cancel)
        except StabilizeError:
            if encoder != "libx264":
                progress(0.4, "Hardware encoder failed; retrying with software encoder…")
                encoder = "libx264"
                kept, skipped = render(ffmpeg, src, tmp, info, W, H, mats, W / aw, encoder,
                                       progress, cancel)
            else:
                raise
        # verify before replacing anything
        out = probe(ffprobe, tmp)
        if info["audio"] and not out["audio"]:
            raise StabilizeError("The source has audio but it could not be carried over; "
                                 "no output was saved.")
        if abs(out["duration"] - info["duration"]) > max(0.5, 2 / info["fps"]):
            raise StabilizeError(f"Output duration {out['duration']:.2f}s doesn't match source "
                                 f"{info['duration']:.2f}s; no output was saved.")
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    note = f"Done ({encoder.replace('h264_', '')})."
    if limited > 0.05:
        note += f" Correction was eased on {limited:.0%} of frames to avoid black edges."
    if skipped:
        note += f" Note: {skipped} audio track(s) ffmpeg can't read (e.g. Apple spatial audio) " \
                f"were left out; {kept} standard track(s) kept."
    progress(1.0, note)
    return note


if __name__ == "__main__":  # command-line use: python stabilize_core.py video.mov [Gentle|Medium|Strong]
    import sys
    s = sys.argv[1]
    print(stabilize(s, output_path(s), sys.argv[2] if len(sys.argv) > 2 else "Gentle",
                    lambda f, m: print(f"{f:5.1%} {m}", end="\r")))
