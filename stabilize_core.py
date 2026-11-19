"""Stabilization engine: ffmpeg decodes/encodes, OpenCV + numpy estimate camera rotation.

Model: a handheld phone shooting a distant scene mostly ROTATES (yaw, pitch, roll).
A pure camera rotation R moves image points by the homography  K R K^-1  (K = lens).

Pass 1: decode small grayscale frames, track corners inside the reference area
        (the bleachers/stands, which you select) with optical flow, turn them into
        3D viewing rays, and solve for the 3D rotation between frames with RANSAC
        (moving people fall out as outliers). The per-frame yaw/pitch/roll path is saved.
Pass 2: smooth that 3D rotation path, then re-render each full-resolution frame as if
        the phone had been held at the smoothed orientation: K * R_smooth * R_actual^T * K^-1.
        A fixed slight zoom hides the edges.
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
    "Gentle": dict(zoom=1.05, sigma=1.0),
    "Medium": dict(zoom=1.07, sigma=1.6),
    "Strong": dict(zoom=1.10, sigma=2.5),
}
ANALYSIS_WIDTH = 800
DEFAULT_F35 = 28.0  # iPhone main camera in video mode, 35mm-equivalent (used if no metadata)
COPYABLE_AUDIO = {"aac"}


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
    f35 = None  # iPhones store the 35mm-equivalent focal length in metadata
    for k, val in list(fmt.get("tags", {}).items()) + list(v.get("tags", {}).items()):
        if "focal_length" in k.lower() and "35" in k:
            try:
                f35 = float(str(val).split()[0])
            except ValueError:
                pass
    return dict(f35=f35 if f35 and 8 <= f35 <= 300 else None, w=w, h=h, fps_str=f"{fr[0]}/{fr[1]}", fps=fr[0] / fr[1], duration=dur,
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


def audio_decodable(ffmpeg, src, index):
    """True if ffmpeg can decode this audio track (iPhone spatial audio/APAC can't be)."""
    r = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", src, "-map", f"0:{index}",
                        "-t", "1", "-f", "null", "-"], capture_output=True)
    return r.returncode == 0


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

def first_frame_png(src, out_png, max_w=960, max_h=600):
    """Write the first frame (orientation applied) for the reference-area picker."""
    ffmpeg, _ = check_dependencies()
    r = subprocess.run([ffmpeg, "-v", "error", "-y", "-nostdin", "-i", src, "-frames:v", "1",
                        "-vf", f"scale={max_w}:{max_h}:force_original_aspect_ratio=decrease",
                        out_png], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_png):
        raise StabilizeError("Couldn't read a frame from this video.\n\n" + r.stderr[-800:])


# ---------------------------------------------------------------- 3D rotation math
def camera_matrix(w, h, f35):
    f = (f35 or DEFAULT_F35) * math.hypot(w, h) / 43.27  # 35mm-equivalent -> pixels
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def kabsch(A, B):
    """Rotation(s) R minimizing |R a - b| for ray sets A, B of shape (..., N, 3)."""
    M = np.swapaxes(A, -1, -2) @ B
    U, _, Vt = np.linalg.svd(M)
    V, Ut = np.swapaxes(Vt, -1, -2).copy(), np.swapaxes(U, -1, -2)
    d = np.sign(np.linalg.det(V @ Ut))
    V[..., :, 2] *= d[..., None]
    return V @ Ut


def rays(pts, Kinv):
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ Kinv.T
    return h / np.linalg.norm(h, axis=1, keepdims=True)


def rv2R(v):
    return cv2.Rodrigues(np.asarray(v, float).reshape(3, 1))[0]


def R2rv(R):
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def rv2q(v):
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.array([1.0, 0, 0, 0])
    return np.concatenate([[math.cos(a / 2)], math.sin(a / 2) * np.asarray(v) / a])


def q2R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def qnorm(Q):
    return Q / np.linalg.norm(Q, axis=-1, keepdims=True)


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


# ---------------------------------------------------------------- pass 1
def estimate_rotation(prev, cur, Kinv, thresh, mask, rng):
    """3D camera rotation R (rays_cur = R @ rays_prev) from tracked points, or None."""
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=600, qualityLevel=0.005,
                                  minDistance=8, blockSize=7, mask=mask)
    if pts is None or len(pts) < 15:
        return None
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **lk)
    fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)
    if ok.sum() < 15:
        return None
    a = rays(pts[ok].reshape(-1, 2).astype(float), Kinv)
    b = rays(nxt[ok].reshape(-1, 2).astype(float), Kinv)
    # RANSAC over 2-point rotation hypotheses (all solved at once)
    idx = rng.integers(0, len(a), (160, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    Rs = kabsch(a[idx], b[idx])
    err = 1.0 - np.einsum("kij,nj,ni->kn", Rs, a, b)
    inl = err[np.argmax((err < thresh).sum(1))] < thresh
    for _ in range(2):
        if inl.sum() < 12:
            return None
        R = kabsch(a[inl], b[inl])
        inl = (1.0 - np.sum((a @ R.T) * b, axis=1)) < thresh
    if inl.sum() < 12 or np.linalg.norm(R2rv(R)) > math.radians(5):
        return None
    return R


def analyze(ffmpeg, src, info, aw, ah, K, region, progress, cancel):
    """Per-frame camera orientation (rotation vectors, frame 0 -> frame t)."""
    dec = Decoder(ffmpeg, src, info["fps_str"], aw, ah, "gray", vf=f"scale={aw}:{ah}:flags=area")
    Kinv = np.linalg.inv(K)
    thresh = 0.5 * (0.7 / K[0, 0]) ** 2  # ~0.7 px angular tolerance
    rng = np.random.default_rng(0)
    box = None
    if region is not None:
        x0, y0, x1, y1 = region
        box = np.array([[x0 * aw, y0 * ah], [x1 * aw, y0 * ah],
                        [x1 * aw, y1 * ah], [x0 * aw, y1 * ah]])
        area0 = (x1 - x0) * aw * (y1 - y0) * ah
    total = max(1, int(round(info["duration"] * info["fps"])))
    path, acc, prev, misses = [], np.eye(3), None, 0
    try:
        while True:
            if cancel.is_set():
                raise Cancelled()
            g = dec.read()
            if g is None:
                break
            if prev is not None:
                mask = None
                if box is not None:  # reference area follows the scene as the camera moves
                    lo = np.clip(box.min(0), 0, [aw, ah]).astype(int)
                    hi = np.clip(box.max(0), 0, [aw, ah]).astype(int)
                    if (hi - lo).prod() >= 0.25 * area0:
                        mask = np.zeros((ah, aw), np.uint8)
                        mask[lo[1]:hi[1], lo[0]:hi[0]] = 255
                R = estimate_rotation(prev, g, Kinv, thresh, mask, rng)
                if R is None and mask is not None:
                    R = estimate_rotation(prev, g, Kinv, thresh, None, rng)
                if R is None:
                    R, misses = np.eye(3), misses + 1
                acc = R @ acc
                if box is not None:
                    hb = np.hstack([box, np.ones((4, 1))]) @ (K @ R @ Kinv).T
                    box = hb[:, :2] / hb[:, 2:]
            path.append(R2rv(acc))
            prev = g
            if len(path) % 15 == 0:
                progress(0.4 * min(1.0, len(path) / total),
                         f"Measuring camera yaw/pitch/roll… frame {len(path)} of ~{total}")
    finally:
        err = dec.close()
    if len(path) < 2:
        raise StabilizeError("Couldn't decode video frames.\n\n" + err)
    return np.array(path)


# ---------------------------------------------------------------- planning
def plan(rotvecs, fps, K, aw, ah, zoom, sigma_s):
    """Per-frame 3x3 homographies (analysis pixels): source frame -> stabilized frame."""
    n = len(rotvecs)
    Kinv = np.linalg.inv(K)
    cx, cy = aw / 2.0, ah / 2.0
    Z = np.array([[zoom, 0, cx - zoom * cx], [0, zoom, cy - zoom * cy], [0, 0, 1.0]])
    corners = np.array([[0, 0, 1], [aw, 0, 1], [aw, ah, 1], [0, ah, 1]], float).T
    Q = np.array([rv2q(v) for v in rotvecs])
    for i in range(1, n):  # keep quaternion signs continuous so averaging is valid
        if Q[i] @ Q[i - 1] < 0:
            Q[i] = -Q[i]
    Rt = [q2R(q) for q in Q]

    def corr(qt, i):  # rotation taking the actual camera to the target camera
        return R2rv(q2R(qt) @ Rt[i].T)

    def fits(c):
        s = np.linalg.inv(Z @ K @ rv2R(c) @ Kinv) @ corners
        s = s[:2] / s[2]
        return bool(np.all((s[0] >= 0) & (s[0] <= aw) & (s[1] >= 0) & (s[1] <= ah)))

    # 1) intentional moves (pans/tilts) vs. held shots, from angular speed
    sm = qnorm(gsmooth(Q, 0.5 * fps))
    dots = np.clip(np.abs(np.sum(sm[1:] * sm[:-1], axis=1)), 0, 1)
    speed = np.append(2 * np.arccos(dots) * fps, 0.0)
    moving = speed > math.radians(2.0)
    for s, e, v in runs(moving):
        if v and e - s < 0.3 * fps:
            moving[s:e] = False
    for s, e, v in runs(moving):
        if not v and e - s < 1.5 * fps:
            moving[s:e] = True

    # 2) target orientation: hold one average orientation during held shots,
    #    follow a smoothed path during deliberate moves
    target = qnorm(gsmooth(Q, sigma_s * fps))
    for s, e, v in runs(moving):
        if v:
            continue
        seg, t = Q[s:e], np.arange(e - s)
        coef = np.polyfit(t, seg, 1)
        cands = [np.repeat(seg.mean(0, keepdims=True), e - s, axis=0),   # fixed framing
                 np.outer(t, coef[0]) + coef[1],                         # slow steady drift
                 gsmooth(seg, 3.0 * fps)]                                # very slow wander
        for c in cands:
            c = qnorm(c)
            if all(fits(corr(c[j], s + j)) for j in range(0, e - s, 2)):
                target[s:e] = c
                break
        else:
            target[s:e] = qnorm(cands[-1])
    target = qnorm(gsmooth(target, 0.3 * fps))

    # 3) never show borders: ease the correction where it exceeds the fixed crop
    cs = [corr(target[i], i) for i in range(n)]
    k_req = np.ones(n)
    for i, c in enumerate(cs):
        if not fits(c):
            lo, hi = 0.0, 1.0
            for _ in range(14):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if fits(c * mid) else (lo, mid)
            k_req[i] = lo
    r = max(1, int(0.5 * fps))
    padded = np.pad(k_req, r, mode="edge")
    k_min = np.min(np.stack([padded[j:j + n] for j in range(2 * r + 1)]), axis=0)
    k = np.minimum(gsmooth(k_min, 0.25 * fps), k_req)
    mats = [Z @ K @ rv2R(cs[i] * k[i]) @ Kinv for i in range(n)]
    return mats, float((k < 0.999).mean())


def render(ffmpeg, src, tmp_out, info, W, H, mats, scale, encoder, progress, cancel):
    crop = f"crop={W}:{H}:0:0" if (W, H) != (info["w"], info["h"]) else None
    dec = Decoder(ffmpeg, src, info["fps_str"], W, H, "bgr24", vf=crop)
    usable = [a for a in info["audio"] if audio_decodable(ffmpeg, src, a["index"])]
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
    S = np.diag([W / scale[0], H / scale[1], 1.0])  # analysis px -> full-res px
    Sinv = np.linalg.inv(S)
    total = len(mats)
    i = 0
    try:
        while True:
            if cancel.is_set():
                raise Cancelled()
            f = dec.read()
            if f is None:
                break
            M = S @ mats[min(i, total - 1)] @ Sinv
            out = cv2.warpPerspective(f, M, (W, H), flags=cv2.INTER_LINEAR,
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

# ---------------------------------------------------------------- outputs & motion cache
def output_path(src):
    return os.path.splitext(src)[0] + "_stabilized.mp4"


MOTION_VERSION = 2


def motion_path(src):
    """Saved camera-rotation analysis, next to the original (e.g. IMG_4569_motion.npz)."""
    return os.path.splitext(src)[0] + "_motion.npz"


def _src_id(src):
    st = os.stat(src)
    return np.array([st.st_size, int(st.st_mtime)], np.int64)


def _region_arr(region):
    return np.array(region if region is not None else [-1, -1, -1, -1], float)


def saved_region(src):
    """Reference area stored with a previous analysis of this video (or None)."""
    try:
        d = np.load(motion_path(src))
        r = d["region"]
        return tuple(float(x) for x in r) if int(d["version"]) == MOTION_VERSION and r[0] >= 0 \
            else None
    except Exception:
        return None


def load_motion(src, info, aw, ah, region):
    try:
        d = np.load(motion_path(src))
        if (int(d["version"]) == MOTION_VERSION and np.array_equal(d["src_id"], _src_id(src))
                and int(d["aw"]) == aw and int(d["ah"]) == ah
                and str(d["fps_str"]) == info["fps_str"]
                and np.allclose(d["region"], _region_arr(region))):
            return d["rotvecs"], d["K"]
    except Exception:
        pass
    return None


def save_motion(src, info, aw, ah, region, rotvecs, K):
    """rotvecs: per-frame camera orientation (Rodrigues vector, radians; x=pitch axis,
    y=yaw axis, z=roll axis) relative to frame 0. K: lens matrix in analysis pixels.
    Any strength can be re-planned from this without re-analyzing."""
    try:
        np.savez(motion_path(src), version=MOTION_VERSION, src_id=_src_id(src), aw=aw, ah=ah,
                 fps_str=info["fps_str"], fps=info["fps"], region=_region_arr(region),
                 rotvecs=rotvecs, K=K)
    except OSError:
        pass  # read-only folder: just analyze again next time


def check_dependencies():
    ff, fp = find_tool("ffmpeg"), find_tool("ffprobe")
    if not ff or not fp:
        raise StabilizeError("ffmpeg was not found.\n\nInstall it in Terminal with:\n"
                             "brew install ffmpeg")
    return ff, fp


def stabilize(src, dst, strength="Gentle", progress=lambda f, m: None, cancel=None,
              region=None):
    """region: reference area (x0, y0, x1, y1) as fractions of the first frame, or None."""
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
    cached = load_motion(src, info, aw, ah, region)
    if cached is None:
        K = camera_matrix(aw, ah, info["f35"])
        rotvecs = analyze(ffmpeg, src, info, aw, ah, K, region, progress, cancel)
        save_motion(src, info, aw, ah, region, rotvecs, K)
    else:
        rotvecs, K = cached
        progress(0.4, "Using saved motion analysis…")
    progress(0.4, "Planning smooth camera orientation…")
    mats, limited = plan(rotvecs, info["fps"], K, aw, ah, cfg["zoom"], cfg["sigma"])

    d, b = os.path.split(dst)
    tmp = os.path.join(d, "." + b + ".partial.mp4")
    try:
        try:
            kept, skipped = render(ffmpeg, src, tmp, info, W, H, mats, (aw, ah), encoder,
                                   progress, cancel)
        except StabilizeError:
            if encoder != "libx264":
                progress(0.4, "Hardware encoder failed; retrying with software encoder…")
                encoder = "libx264"
                kept, skipped = render(ffmpeg, src, tmp, info, W, H, mats, (aw, ah), encoder,
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
