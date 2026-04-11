"""
Matplotlib-based local live visualisation backend.

Uses ``FuncAnimation`` with a Tkinter window to display neuron
activations in real-time as a scatter plot with glow-mapped colours,
plus interactive speed/pause controls via matplotlib widgets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sdmn.visualization.live_visualizer import LiveVisualizer

_CLASS_BASE_COLORS = {
    "sensory":     np.array([0.29, 0.62, 1.00]),
    "interneuron": np.array([0.30, 0.69, 0.31]),
    "motor":       np.array([1.00, 0.60, 0.00]),
    "unknown":     np.array([0.62, 0.62, 0.62]),
}


class MatplotlibVisualizer:
    """
    Local fallback renderer using matplotlib.

    Opened by ``LiveVisualizer.run(backend="matplotlib")``.
    """

    def __init__(self, controller: "LiveVisualizer"):
        self.ctrl = controller

    def show(self, figsize=(12, 9), interval_ms=33, **_kw) -> None:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Slider, Button

        meta = self.ctrl.get_positions_list()
        n = len(meta)
        ids = [m[0] for m in meta]
        xs = np.array([m[1] for m in meta])
        ys = np.array([m[2] for m in meta])
        classes = [m[3] for m in meta]
        base_colors = np.array([_CLASS_BASE_COLORS.get(c, _CLASS_BASE_COLORS["unknown"]) for c in classes])

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0a0a14")
        ax.set_facecolor("#0a0a14")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.axis("off")

        scatter = ax.scatter(xs, ys, s=40, c=base_colors * 0.15, edgecolors="none", zorder=3)

        edge_data = self.ctrl.get_edges()
        id_idx = {nid: i for i, nid in enumerate(ids)}
        edge_lines = []
        for src, dst in edge_data:
            si, di = id_idx.get(src), id_idx.get(dst)
            if si is not None and di is not None:
                ln, = ax.plot([xs[si], xs[di]], [ys[si], ys[di]],
                              color=(0.3, 0.5, 0.8, 0.03), linewidth=0.3, zorder=1)
                edge_lines.append((si, di, ln))

        title = ax.set_title("0.00 ms | step 0", color="#aaeeff", fontsize=12, pad=10)

        plt.subplots_adjust(bottom=0.15)

        ax_speed = plt.axes([0.2, 0.04, 0.35, 0.025], facecolor="#1a1a2e")
        slider = Slider(ax_speed, "Speed", -2, 2, valinit=np.log10(self.ctrl.speed),
                        color="#4a9eff")

        def on_speed(val):
            self.ctrl.set_speed(10 ** val)
        slider.on_changed(on_speed)

        ax_pause = plt.axes([0.65, 0.03, 0.08, 0.04])
        btn_pause = Button(ax_pause, "Pause", color="#1a1a2e", hovercolor="#252540")

        def on_pause(_ev):
            if self.ctrl.is_paused:
                self.ctrl.resume()
                btn_pause.label.set_text("Pause")
            else:
                self.ctrl.pause()
                btn_pause.label.set_text("Resume")
        btn_pause.on_clicked(on_pause)

        ax_step = plt.axes([0.75, 0.03, 0.06, 0.04])
        btn_step = Button(ax_step, "Step", color="#1a1a2e", hovercolor="#252540")
        btn_step.on_clicked(lambda _: self.ctrl.step_once())

        def update(_frame_num):
            frame = self.ctrl.get_latest_frame()
            if frame is None:
                return (scatter,)

            bri = np.zeros(n)
            for i, nid in enumerate(ids):
                v = frame.voltages.get(nid, -65.0)
                bri[i] = max(0.0, min(1.0, (v - (-65.0)) / ((-20.0) - (-65.0))))

            colors = base_colors * (0.15 + bri[:, None] * 0.85)
            colors = np.clip(colors, 0, 1)
            scatter.set_facecolors(colors)
            scatter.set_sizes(40 + bri * 120)

            title.set_text(f"{frame.time_ms:.2f} ms | step {frame.step}")

            return (scatter,)

        _anim = FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)
        plt.show()
