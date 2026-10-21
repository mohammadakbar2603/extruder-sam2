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
from IPython.display import display

from .calib import POINT_LABELS, POINT_NAMES
from .video import read_frame
from .viz import POINT_COLOURS


class ReferenceClicker:
    def __init__(self, video, info, clean_image, save_path, points=None, view_centre=(722, 488), view_half=260):
        self.video, self.info, self.clean = str(video), info, clean_image
        self.save_path = Path(save_path)
        self.points = {k: (tuple(v) if v is not None else None) for k, v in (points or {}).items()}
        self._read = lru_cache(maxsize=32)(lambda i: read_frame(self.video, i, info["fps"], info["width"], info["height"]))

        self.w_point = W.Dropdown(options=[(f"{POINT_LABELS[n]}", n) for n in POINT_NAMES], description="Point:",
                                  layout=W.Layout(width="420px"))
        self.w_view = W.ToggleButtons(options=["Clean die (median)", "Single frame"], description="View:")
        self.w_frame = W.IntSlider(0, 0, info["n_frames"] - 1, description="Frame:", continuous_update=False,
                                   layout=W.Layout(width="600px"))
        self.w_gain = W.FloatSlider(1.0, min=1.0, max=3.0, step=0.1, description="Brightness:", continuous_update=False)
        self.w_clear = W.Button(description="Remove this point", icon="trash")
        self.w_save = W.Button(description="Save points", button_style="success", icon="save")
        self.w_msg = W.Output()

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
                        W.HBox([self.w_view, self.w_gain]), self.w_frame, self.fig.canvas, self.w_msg]))
        self._say("Pick a point in the list, then left-click it in the picture.")

    def _say(self, text):
        with self.w_msg:
            self.w_msg.clear_output()
            print(text)
            print("Current points:", {k: (None if v is None else (round(v[0], 1), round(v[1], 1)))
                                      for k, v in self.points.items()})

    def _refresh_image(self):
        img = self.clean if self.w_view.value.startswith("Clean") else self._read(self.w_frame.value)
        g = self.w_gain.value
        if g != 1.0:
            img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
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
        data = {"video": Path(self.video).name, "clicked": {k: v for k, v in self.points.items() if v is not None}}
        old = json.loads(self.save_path.read_text()) if self.save_path.exists() else {}
        old.update(data)
        self.save_path.write_text(json.dumps(old, indent=1))
        self._say(f"Saved to {self.save_path}. You can run the next cell now.")
