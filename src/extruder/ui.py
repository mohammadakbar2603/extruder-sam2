"""Click tool for the notebook: mark hole A and the reference points on the die plate.

Works in VS Code and JupyterLab with the interactive matplotlib backend (ipympl):
run `%matplotlib widget` before creating the tool.

How to use
----------
* Choose what you are clicking in the "Point" list, then left-click on it in the picture.
* Use the magnifier button in the figure toolbar to zoom in for an exact click;
  click the magnifier again to switch zoom off before clicking a point
  (clicks are ignored while zoom or pan is active).
* "Clean die" view = the median of many frames: moving product disappears, so the
  bolts are easy to see. Switch to "Single frame" and use the slider to see a real frame.
* Press "Save points" when done. The points are written to clicks.json and reused the
  next time, so you only click once per video.
"""
import json
from functools import lru_cache
from pathlib import Path

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import display

from .calib import POINT_LABELS, POINT_NAMES
from .video import apply_brightness, auto_brightness, read_frame
from .viz import POINT_COLOURS


class ReferenceClicker:
    def __init__(self, video, info, clean_image, save_path, points=None, view_centre=(722, 488), view_half=260,
                 brightness=1.0):
        self.video, self.info, self.clean = str(video), info, clean_image
        self.save_path = Path(save_path)
        self.points = {k: (tuple(v) if v is not None else None) for k, v in (points or {}).items()}
        self._read = lru_cache(maxsize=32)(lambda i: read_frame(self.video, i, info["fps"], info["width"], info["height"]))

        self.w_point = W.Dropdown(options=[(f"{POINT_LABELS[n]}", n) for n in POINT_NAMES], description="Point:",
                                  layout=W.Layout(width="420px"))
        self.w_view = W.ToggleButtons(options=["Clean die (median)", "Single frame"], description="View:")
        self.w_frame = W.IntSlider(0, 0, info["n_frames"] - 1, description="Frame:", continuous_update=False,
                                   layout=W.Layout(width="600px"))
        self.w_gain = W.FloatSlider(1.0, min=1.0, max=3.0, step=0.05, description="Brightness:",
                                    continuous_update=False, tooltip="saved with the points; SAM 2 sees the video this bright")
        self.w_clear = W.Button(description="Remove this point", icon="trash")
        self.w_save = W.Button(description="Save points", button_style="success", icon="save")
        self.w_auto = W.Button(description="Auto brightness", icon="sun")
        self.w_msg = W.Output()
        self.w_gain.value = float(brightness or 1.0)
        self.w_auto.on_click(self._on_auto)

        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(10, 6.2))
        plt.ion()
        self.fig.canvas.header_visible = False
        self.im = self.ax.imshow(self.clean)
        cx, cy = self.points.get("hole_A") or view_centre
        self.ax.set_xlim(cx - view_half * 1.4, cx + view_half * 1.4)
        self.ax.set_ylim(cy + view_half, cy - view_half)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.marks = []
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        for w in (self.w_view, self.w_frame, self.w_gain):
            w.observe(lambda ch: self._refresh_image(), names="value")
        self.w_clear.on_click(self._on_clear)
        self.w_save.on_click(self._on_save)
        self._draw_points()

    # -------------------------------------------------------------- display
    def show(self):
        display(W.VBox([W.HBox([self.w_point, self.w_clear, self.w_save]),
                        W.HBox([self.w_view, self.w_gain, self.w_auto]), self.w_frame, self.fig.canvas, self.w_msg]))
        self._refresh_image()
        self._say("Pick a point in the list, then left-click it in the picture.")

    @property
    def brightness(self):
        return float(self.w_gain.value)

    def _on_auto(self, _):
        hole = self.points.get("hole_A")
        if hole is None:
            self._say("Click hole A first: auto brightness looks at the area around it.")
            return
        self.w_gain.value = auto_brightness(self.clean, hole)
        self._say(f"Brightness set to {self.w_gain.value:.2f}. Press Save points to keep it.")

    def _say(self, text):
        with self.w_msg:
            self.w_msg.clear_output()
            print(text)
            print("Current points:", {k: (None if v is None else (round(v[0], 1), round(v[1], 1)))
                                      for k, v in self.points.items()})

    def _refresh_image(self):
        img = self.clean if self.w_view.value.startswith("Clean") else self._read(self.w_frame.value)
        img = apply_brightness(img, self.w_gain.value)
        self.im.set_data(img)
        self.fig.canvas.draw_idle()

    def _draw_points(self):
        for m in self.marks:
            m.remove()
        self.marks = []
        for name, p in self.points.items():
            if p is None:
                continue
            c = POINT_COLOURS.get(name, "magenta")
            self.marks += self.ax.plot(*p, "+", color=c, ms=16, mew=2)
            self.marks.append(self.ax.annotate(name, p, xytext=(8, -8), textcoords="offset points", color=c,
                                               fontsize=9, weight="bold"))
        self.fig.canvas.draw_idle()

    # -------------------------------------------------------------- events
    def _on_click(self, ev):
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", "") != "":
            return  # zoom or pan is active
        if ev.inaxes is not self.ax or ev.button != 1 or ev.xdata is None:
            return
        name = self.w_point.value
        self.points[name] = (float(ev.xdata), float(ev.ydata))
        self._draw_points()
        missing = [n for n in POINT_NAMES if self.points.get(n) is None]
        if missing:
            self.w_point.value = missing[0]
        self._say(f"Set {name} at ({ev.xdata:.1f}, {ev.ydata:.1f}).")

    def _on_clear(self, _):
        self.points.pop(self.w_point.value, None)
        self._draw_points()
        self._say(f"Removed {self.w_point.value}.")

    def _on_save(self, _):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"video": Path(self.video).name, "clicked": {k: v for k, v in self.points.items() if v is not None},
                "brightness": self.brightness}
        old = json.loads(self.save_path.read_text()) if self.save_path.exists() else {}
        old.update(data)
        self.save_path.write_text(json.dumps(old, indent=1))
        self._say(f"Saved to {self.save_path}. You can run the next cell now.")


class MultiVideoClicker:
    """Click tool for many videos: pick a video, click its points, save. Saved once, re-used forever.

    * "Video": choose the video (a check mark = points already saved).
    * "Point": what you are about to click; then left-click it in the picture.
    * "Copy points from previous video": the camera usually did not move between clips of
      one setting, so start from the previous video's points and only correct what moved.
    * "Save": writes results/<video>/clicks.json. "Save & next" also moves to the next video.
    Clean-die pictures must exist already (results/<video>/clean_die.png).
    """

    def __init__(self, videos, results_dir, info_of, view_half=260, required=("hole_A",)):
        self.videos = [Path(v) for v in videos]
        self.results_dir = Path(results_dir)
        self.info_of = info_of            # function: video path -> probe() dict
        self.required = required
        self.points = {}
        self.view_half = view_half
        self.w_video = W.Dropdown(description="Video:", layout=W.Layout(width="330px"))
        self.w_point = W.Dropdown(options=[(f"{POINT_LABELS[n]}", n) for n in POINT_NAMES], description="Point:",
                                  layout=W.Layout(width="420px"))
        self.w_view = W.ToggleButtons(options=["Clean die (median)", "Single frame"], description="View:")
        self.w_frame = W.IntSlider(0, 0, 1, description="Frame:", continuous_update=False, layout=W.Layout(width="600px"))
        self.w_gain = W.FloatSlider(1.0, min=1.0, max=3.0, step=0.05, description="Brightness:",
                                    continuous_update=False, tooltip="saved with the points; SAM 2 sees the video this bright")
        self.w_copy = W.Button(description="Copy points from previous video", icon="copy", layout=W.Layout(width="260px"))
        self.w_clear = W.Button(description="Remove this point", icon="trash")
        self.w_save = W.Button(description="Save", button_style="success", icon="save")
        self.w_next = W.Button(description="Save & next", button_style="info", icon="arrow-right")
        self.w_auto = W.Button(description="Auto brightness", icon="sun")
        self.w_msg = W.Output()
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(10, 6.2))
        plt.ion()
        self.fig.canvas.header_visible = False
        self.im = self.ax.imshow(np.zeros((10, 10, 3), np.uint8))
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.marks = []
        self._frames = {}
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        for w in (self.w_view, self.w_frame, self.w_gain):
            w.observe(lambda ch: self._refresh_image(), names="value")
        self.w_video.observe(lambda ch: self._load_video(), names="value")
        self.w_copy.on_click(self._on_copy)
        self.w_clear.on_click(self._on_clear)
        self.w_save.on_click(lambda _: self._save())
        self.w_next.on_click(self._on_next)
        self.w_auto.on_click(self._on_auto)
        self._update_video_list(select=next((v for v in self.videos if not self.has_clicks(v)), self.videos[0]))

    # ---------------------------------------------------------- saved state
    def clicks_path(self, video):
        return self.results_dir / Path(video).stem / "clicks.json"

    def saved_points(self, video):
        p = self.clicks_path(video)
        return json.loads(p.read_text()).get("clicked", {}) if p.exists() else {}

    def saved_brightness(self, video):
        p = self.clicks_path(video)
        return float(json.loads(p.read_text()).get("brightness", 1.0)) if p.exists() else 1.0

    def has_clicks(self, video):
        pts = self.saved_points(video)
        return all(pts.get(r) is not None for r in self.required)

    def status(self):
        """Table: which points are saved for which video."""
        import pandas as pd
        rows = []
        for v in self.videos:
            pts = self.saved_points(v)
            rows.append(dict(video=v.name, **{n: ("yes" if pts.get(n) is not None else "") for n in POINT_NAMES},
                             brightness=self.saved_brightness(v), ready="yes" if self.has_clicks(v) else "NO"))
        return pd.DataFrame(rows)

    # ---------------------------------------------------------- display
    def show(self):
        display(W.VBox([W.HBox([self.w_video, self.w_copy]),
                        W.HBox([self.w_point, self.w_clear, self.w_save, self.w_next]),
                        W.HBox([self.w_view, self.w_gain, self.w_auto]), self.w_frame, self.fig.canvas, self.w_msg]))
        self._say("Pick a point type, then left-click it in the picture.")

    def _update_video_list(self, select=None):
        opts = [((("✓ " if self.has_clicks(v) else "   ") + v.name), str(v)) for v in self.videos]
        cur = str(select) if select is not None else self.w_video.value
        self.w_video.options = opts
        self.w_video.value = cur

    def _video(self):
        return Path(self.w_video.value)

    def _load_video(self):
        v = self._video()
        self.info = self.info_of(v)
        self.w_frame.max = self.info["n_frames"] - 1
        clean_path = self.results_dir / v.stem / "clean_die.png"
        self.clean = np.asarray(Image.open(clean_path).convert("RGB")) if clean_path.exists() else \
            read_frame(str(v), 0, self.info["fps"], self.info["width"], self.info["height"])
        self.points = {k: tuple(p) for k, p in self.saved_points(v).items() if p is not None}
        if self.clicks_path(v).exists():
            self.w_gain.value = self.saved_brightness(v)   # a new video keeps the current setting
        self._refresh_image()
        cx, cy = self.points.get("hole_A") or (self.info["width"] / 2, self.info["height"] / 2)
        h = self.view_half
        self.ax.set_xlim(cx - h * 1.4, cx + h * 1.4); self.ax.set_ylim(cy + h, cy - h)
        self._draw_points()
        self._say(f"{v.name}: " + ("points loaded from the saved file." if self.points else "no points yet."))

    def _frame(self, i):
        key = (str(self._video()), i)
        if key not in self._frames:
            if len(self._frames) > 16:
                self._frames.clear()
            self._frames[key] = read_frame(str(self._video()), i, self.info["fps"], self.info["width"], self.info["height"])
        return self._frames[key]

    def _refresh_image(self):
        img = self.clean if self.w_view.value.startswith("Clean") else self._frame(self.w_frame.value)
        img = apply_brightness(img, self.w_gain.value)
        self.im.set_data(img)
        self.im.set_extent((0, img.shape[1], img.shape[0], 0))
        self.fig.canvas.draw_idle()

    def _draw_points(self):
        for m in self.marks:
            m.remove()
        self.marks = []
        for name, p in self.points.items():
            c = POINT_COLOURS.get(name, "magenta")
            self.marks += self.ax.plot(*p, "+", color=c, ms=16, mew=2)
            self.marks.append(self.ax.annotate(name, p, xytext=(8, -8), textcoords="offset points", color=c,
                                               fontsize=9, weight="bold"))
        self.fig.canvas.draw_idle()

    def _say(self, text):
        with self.w_msg:
            self.w_msg.clear_output()
            print(text)
            print("Points:", {k: (round(v[0], 1), round(v[1], 1)) for k, v in self.points.items()},
                  f"| brightness {self.w_gain.value:.2f}")
            done = sum(self.has_clicks(v) for v in self.videos)
            print(f"Videos with saved points: {done} of {len(self.videos)}")

    # ---------------------------------------------------------- events
    def _on_click(self, ev):
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", "") != "":
            return
        if ev.inaxes is not self.ax or ev.button != 1 or ev.xdata is None:
            return
        name = self.w_point.value
        self.points[name] = (float(ev.xdata), float(ev.ydata))
        self._draw_points()
        self._say(f"Set {name} at ({ev.xdata:.1f}, {ev.ydata:.1f}). Press Save when done.")

    def _on_clear(self, _):
        self.points.pop(self.w_point.value, None)
        self._draw_points()
        self._say(f"Removed {self.w_point.value}.")

    def _on_auto(self, _):
        hole = self.points.get("hole_A")
        if hole is None:
            self._say("Click hole A first: auto brightness looks at the area around it.")
            return
        self.w_gain.value = auto_brightness(self.clean, hole)
        self._say(f"Brightness set to {self.w_gain.value:.2f}. Press Save to keep it.")

    def _on_copy(self, _):
        i = self.videos.index(self._video())
        prev = next((v for v in reversed(self.videos[:i]) if self.saved_points(v)), None)
        if prev is None:
            self._say("No earlier video has saved points.")
            return
        self.points = {k: tuple(p) for k, p in self.saved_points(prev).items()}
        self.w_gain.value = self.saved_brightness(prev)
        self._draw_points()
        self._say(f"Copied the points of {prev.name}. Check them (zoom in), move any that are off, then Save.")

    def _save(self):
        v = self._video()
        missing = [r for r in self.required if r not in self.points]
        if missing:
            self._say(f"Not saved: please click {missing} first.")
            return False
        p = self.clicks_path(v)
        p.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(p.read_text()) if p.exists() else {}
        if old.get("clicked") != {k: list(map(float, xy)) for k, xy in self.points.items()}:
            old.pop("refined", None)        # clicks changed: refined positions must be recomputed
        old.update(video=v.name, clicked={k: list(map(float, xy)) for k, xy in self.points.items()},
                   brightness=float(self.w_gain.value))
        p.write_text(json.dumps(old, indent=1))
        self._update_video_list(select=v)
        self._say(f"Saved {p}.")
        return True

    def _on_next(self, _):
        if not self._save():
            return
        i = self.videos.index(self._video())
        nxt = next((v for v in self.videos[i + 1:] + self.videos[:i] if not self.has_clicks(v)), None)
        if nxt is None:
            self._say("All videos have saved points. You can run the next cell.")
            return
        self._update_video_list(select=nxt)
