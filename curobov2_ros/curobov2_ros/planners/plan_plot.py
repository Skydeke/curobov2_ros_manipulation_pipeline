#!/usr/bin/env python3
"""
Motion-plan debug image rendering (shared by the open-loop planners).

Turns a planned joint trajectory into an RGB image (pos / vel / acc / jerk
panels) matching the interactive viser GUI's trajectory plot, so the same
debug image can be published to ROS without materializing any GUI.

Pure function on numpy arrays (no ROS, no torch): the callback that owns the
trajectory is responsible for collapsing batched ``[B, T, D]`` data down to a
``[T, D]`` per-waypoint layout before calling ``render_plan_plot``.
"""

import numpy as np


def render_plan_plot(
    position,
    names,
    dt,
    velocity=None,
    acceleration=None,
    title="",
):
    """Render a multi-panel joint trajectory plot to an RGB image.

    Mirrors the viser GUI's stacked Position / Velocity / Accel / Jerk panels
    using the real joint names. position/velocity/acceleration are ``[T, D]``
    (one row per waypoint); missing velocity/acceleration are finite-differenced
    from position. Returns an ``H x W x 3`` uint8 numpy array (RGB).

    matplotlib is imported lazily so this stays an optional feature: when it is
    unavailable the caller (see SinglePlanner) simply skips publishing.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    pos = np.asarray(position, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    n, dof = pos.shape

    if velocity is not None and len(velocity):
        vel = np.asarray(velocity, dtype=float)
        if vel.ndim == 1:
            vel = vel.reshape(1, -1)
        vel = vel[:n, :dof]
    else:
        vel = np.zeros((n, dof))

    d = max(float(dt), 1e-6)
    if not np.any(vel):
        vel = np.diff(pos, axis=0, prepend=pos[:1]) / d
    acc = np.diff(vel, axis=0, prepend=vel[:1]) / d
    jrk = np.diff(acc, axis=0, prepend=acc[:1]) / d

    names = list(names or [f"J{i}" for i in range(dof)])
    plot_data = [
        (pos, "Position (rad)"),
        (vel, "Velocity (rad/s)"),
        (acc, "Accel (rad/s^2)"),
        (jrk, "Jerk (rad/s^3)"),
    ]

    n_plots = len(plot_data)
    fig, axes = plt.subplots(
        n_plots, 1, figsize=(6, 2 * n_plots), dpi=100, sharex=True
    )
    if n_plots == 1:
        axes = [axes]
    t = np.arange(n) * d

    for ax, (data, ylabel) in zip(axes, plot_data):
        for j in range(dof):
            label = names[j] if j < len(names) else f"J{j}"
            if len(label) > 10:
                label = label[:8] + ".."
            ax.plot(t, data[:, j], linewidth=1.2, label=label)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return np.array(PILImage.open(buf).convert("RGB"))
