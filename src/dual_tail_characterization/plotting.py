import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def amplitude_limits(w):
    low = min(w.final.min(), min(w.config.levels))
    high = max(w.final.max(), max(w.config.levels))
    margin = .08 * (high - low)
    return low - margin, high + margin


def plot_waveform(w, path, start_ui=0, count_ui=16):
    c = w.config
    end = min(start_ui + count_ui, c.n_symbols)
    mask = (w.time >= start_ui * c.symbol_period) & (w.time <= end * c.symbol_period)
    fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
    for level in c.levels:
        ax.axhline(level, color="gray", lw=.6, alpha=.5)
    for k in range(start_ui, end + 1):
        ax.axvline(k, color="gray", lw=.5, alpha=.4)
        if k < end:
            ax.text(k + .5, .96, str(w.symbol_indices[k]), ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1))
    ax.step(w.time[mask] / c.symbol_period, w.ideal[mask], where="post", label="Ideal", color="gray", alpha=.7)
    ax.plot(w.time[mask] / c.symbol_period, w.final[mask], label="Final", lw=1)
    ax.set(xlabel="Time (UI)", ylabel=f"Amplitude ({c.amplitude_unit})", xlim=(start_ui, end), ylim=amplitude_limits(w), title="Continuous PAM waveform · transmitted indices above")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def eye_histogram(eye, limits, amplitude_bins=240):
    # Each numerical sample contributes exactly once; no image smoothing.
    xedges = np.linspace(0, eye.relative_time_ui[-1], len(eye.relative_time_ui) + 1)
    yedges = np.linspace(*limits, amplitude_bins + 1)
    counts = np.zeros((len(xedges) - 1, amplitude_bins), dtype=np.int64)
    for column, values in enumerate(eye.traces.T):
        counts[column] = np.histogram(values, bins=yedges)[0]
    return counts, xedges, yedges


def plot_eye(eye, w, path, mode="density"):
    limits = amplitude_limits(w)
    counts, xedges, yedges = eye_histogram(eye, limits)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    if mode == "density":
        mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_equal(counts.T, 0),
                             norm=LogNorm(vmin=1, vmax=max(2, int(counts.max()))), cmap="viridis", rasterized=True)
        fig.colorbar(mesh, ax=ax, label="Sample count (log color scale)")
    elif mode == "overlay":
        # Separate columns are separate line artists: no trace-to-trace joins.
        ax.plot(eye.relative_time_ui, eye.traces[:500].T, color="tab:blue", alpha=.06, lw=.6)
    else:
        raise ValueError("plot mode must be density or overlay")
    for level in w.config.levels:
        ax.axhline(level, color="gray", lw=.5, alpha=.5)
    ax.set(xlabel="Relative time (UI)", ylabel=f"Amplitude ({w.config.amplitude_unit})",
           xlim=(0, eye.relative_time_ui[-1]), ylim=limits,
           title=f"{int(eye.relative_time_ui[-1])}UI eye · phase {w.config.display_phase_ui:g} UI · {len(eye.traces):,} traces")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return counts, xedges, yedges
