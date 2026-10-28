"""Reading the camera videos.

The DJI camera records 10-bit HEVC at about 240 frames per second, which many Python
video readers cannot open. We therefore call the `ffmpeg` program (installed in the
conda environment) and read its raw RGB output.

Frame numbers are 0-based everywhere in this project: frame 0 is the first frame.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


def _tool(name):
    """Path to ffmpeg / ffprobe: the copy inside this conda environment if present, else the system one."""
    env = Path(sys.executable).parent
    for folder in (env, env / "Library" / "bin"):   # Linux/macOS, Windows
        for cand in (folder / name, folder / f"{name}.exe"):
            if cand.exists():
                return str(cand)
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(f"{name} not found. Install it with: conda install -c conda-forge ffmpeg")
    return found


def list_videos(folder):
    """All video files in a folder (not recursive), sorted by name."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def probe(path):
    """Basic facts about a video: frames per second, number of frames, size, duration."""
    out = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,nb_frames,width,height,codec_name,pix_fmt",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    st = info["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(info["format"]["duration"])
    n = int(st.get("nb_frames") or round(duration * fps))
    return dict(path=str(path), fps=fps, n_frames=n, width=int(st["width"]), height=int(st["height"]),
                duration_s=duration, codec=st.get("codec_name"), pixel_format=st.get("pix_fmt"))


def read_frame(path, index, fps, width=None, height=None):
    """One full frame (H, W, 3) RGB uint8, by frame number.

    Seeks to just before the frame's timestamp, so it is fast even late in the clip.
    """
    if width is None or height is None:
        info = probe(path)
        width, height = info["width"], info["height"]
    t = max(0.0, (index - 0.25) / fps)
    cmd = [_tool("ffmpeg"), "-v", "error", "-ss", f"{t:.6f}", "-i", str(path), "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()


BRIGHTNESS_TARGET = 0.22   # "auto" brightness: median grey level around hole A (0-1) of the brighter videos


def apply_brightness(img, brightness=1.0):
    """Brighten an RGB uint8 image: out = in ^ (1 / brightness), on a 0-1 scale (a "gamma" curve).

    1.0 = unchanged; 1.5 or 2.0 = brighter. Dark parts are lifted much more than bright ones,
    so bright metal does not burn out. The video processing uses exactly the same curve.
    """
    if brightness is None or abs(brightness - 1.0) < 1e-6:
        return img
    lut = (255.0 * (np.arange(256) / 255.0) ** (1.0 / brightness) + 0.5).astype(np.uint8)
    return lut[np.asarray(img)]


def auto_brightness(clean_image, hole_xy, half=180, target=BRIGHTNESS_TARGET, max_value=3.0):
    """Brightness that makes the area around hole A as bright as `target` (never darkens)."""
    g = np.asarray(clean_image).astype(np.float32).mean(-1) / 255.0
    x, y = int(round(hole_xy[0])), int(round(hole_xy[1]))
    m = float(np.median(g[max(0, y - half):y + half, max(0, x - half):x + half]))
    if m <= 0 or m >= target:
        return 1.0
    return float(np.clip(round(np.log(m) / np.log(target) / 0.05) * 0.05, 1.0, max_value))


def iter_crops(path, x0, y0, size, width=None, height=None, brightness=1.0):
    """Yield every frame of a square crop (size x size) whose top-left corner is (x0, y0).

    Streams the video, so memory use stays small even for long clips. If the crop
    reaches outside the frame it is padded with black. `brightness` > 1 lifts dark parts
    (see apply_brightness); it is applied at 16 bits per colour, before rounding to 8 bits,
    so the camera's 10-bit shadow detail is kept.
    """
    if width is None or height is None:
        info = probe(path)
        width, height = info["width"], info["height"]
    # ffmpeg can only crop inside the frame: crop the valid part, then pad
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(width, x0 + size), min(height, y0 + size)
    # convert to RGB *before* cropping: the camera stores colour at half resolution, and cropping
    # that directly makes ffmpeg silently move odd x/y positions by 1 px
    vf = f"format=rgb24,crop={cx1 - cx0}:{cy1 - cy0}:{cx0}:{cy0}"
    if brightness is not None and abs(brightness - 1.0) > 1e-6:
        e = 1.0 / brightness
        curve = f"maxval*pow(val/maxval\\,{e:.6f})"
        vf = (f"format=rgb48le,lutrgb=r={curve}:g={curve}:b={curve},format=rgb24,"
              f"crop={cx1 - cx0}:{cy1 - cy0}:{cx0}:{cy0}")
    if (cx0, cy0, cx1, cy1) != (x0, y0, x0 + size, y0 + size):
        vf += f",pad={size}:{size}:{cx0 - x0}:{cy0 - y0}:black"
    cmd = [_tool("ffmpeg"), "-v", "error", "-i", str(path), "-vf", vf, "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    nbytes = size * size * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=nbytes * 8)
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(size, size, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def read_crops(path, x0, y0, size, brightness=1.0):
    """All frames of a square crop as one array (n, size, size, 3). Use for small crops only."""
    return np.stack(list(iter_crops(path, x0, y0, size, brightness=brightness)))


def clean_die_image(path, n_samples=70):
    """A "clean die" picture: the per-pixel median of frames spread over the whole clip.

    Product and knife move, the die plate does not, so the median keeps the plate
    (bolts and holes) and mostly removes the moving pieces. Used for clicking the
    reference points. Takes about 15 s for a 10 s clip.
    """
    info = probe(path)
    step = max(1, info["n_frames"] // n_samples)
    cmd = [_tool("ffmpeg"), "-v", "error", "-i", str(path), "-vf", f"select='not(mod(n\\,{step}))'",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, info["height"], info["width"], 3)
    # median in horizontal strips to keep memory moderate
    out = np.empty(frames.shape[1:], np.uint8)
    for r in range(0, info["height"], 120):
        out[r:r + 120] = np.median(frames[:, r:r + 120], axis=0).astype(np.uint8)
    return out
