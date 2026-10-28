import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import numbers


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
    if w.symbol_endpoint_values is not None:
        selected = (w.symbol_centers >= start_ui*c.symbol_period) & (w.symbol_centers <= end*c.symbol_period)
        ax.scatter(w.symbol_centers[selected]/c.symbol_period, w.symbol_endpoint_values[selected],
                   s=14, color="tab:orange", zorder=3, label="Shared symbol-center endpoints")
    ax.set(xlabel="Time (UI)", ylabel=f"Amplitude ({c.amplitude_unit})", xlim=(start_ui, end), ylim=amplitude_limits(w), title="Continuous PAM waveform · transmitted indices above")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def eye_histogram(eye, limits, amplitude_bins=64, time_bins=None):
    # Each numerical sample contributes exactly once; no image smoothing.
    if time_bins is None:
        time_bins = len(eye.relative_time_ui)
    for name, value in (("amplitude_bins", amplitude_bins), ("time_bins", time_bins)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if time_bins > len(eye.relative_time_ui):
        raise ValueError("time_bins must not exceed the number of sample phases")
    xedges = np.linspace(0, eye.relative_time_ui[-1], time_bins + 1)
    yedges = np.linspace(*limits, amplitude_bins + 1)
    counts = np.zeros((time_bins, amplitude_bins), dtype=np.int64)
    time_indices = np.minimum(np.searchsorted(xedges, eye.relative_time_ui, side="right") - 1, time_bins - 1)
    for column, values in enumerate(eye.traces.T):
        counts[time_indices[column]] += np.histogram(values, bins=yedges)[0]
    return counts, xedges, yedges


def plot_eye(eye, w, path, mode="density", amplitude_bins=64, time_bins=None, reference_lines=True):
    limits = amplitude_limits(w)
    counts, xedges, yedges = eye_histogram(eye, limits, amplitude_bins, time_bins)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    if mode == "density":
        ax.set_facecolor("black")
        # Display-only source line segments bridge deterministic sample phases.
        # These lines never contribute to counts or their exported bin edges.
        ax.plot(eye.relative_time_ui, eye.traces[:500].T, color="white", alpha=.035, lw=.6)
        mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_equal(counts.T, 0),
                             norm=LogNorm(vmin=1, vmax=max(2, int(counts.max()))), cmap="gray", rasterized=True)
        fig.colorbar(mesh, ax=ax, label="Sample count (log color scale)")
    elif mode == "overlay":
        # Separate columns are separate line artists: no trace-to-trace joins.
        ax.plot(eye.relative_time_ui, eye.traces[:500].T, color="tab:blue", alpha=.06, lw=.6)
    else:
        raise ValueError("plot mode must be density or overlay")
    for level in w.config.levels:
        ax.axhline(level, color="gray", lw=.5, alpha=.5)
    length = int(eye.relative_time_ui[-1])
    if reference_lines:
        references = [.5] if length == 1 else [.5, 1., 1.5] if length == 2 else np.arange(.5, length, .5)
        for position in references:
            ax.axvline(position - w.config.display_phase_ui, color="gray", lw=.5, alpha=.6, linestyle="--")
    name = "1UI transition pattern" if length == 1 else f"{length}UI eye diagram"
    ax.set(xlabel="Relative time (UI)", ylabel=f"Amplitude ({w.config.amplitude_unit})",
           xlim=(0, eye.relative_time_ui[-1]), ylim=limits,
           title=f"{name} · phase {w.config.display_phase_ui:g} UI · {min(500, len(eye.traces)) if mode == 'overlay' else len(eye.traces):,} traces"
                 + ("\nFaint lines: first 500 source traces (display only)" if mode == "density" else ""))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return counts, xedges, yedges
