import json
from pathlib import Path
import platform
import numpy as np
import scipy
import matplotlib
from .eyes import paired_eyes
from .plotting import plot_waveform, plot_eye


def save_run(w, output, plot_mode="density", description="", csv_interval_ui=None,
             amplitude_bins=64, time_bins_per_ui=None, reference_lines=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    one, two = paired_eyes(w)
    if not len(two.traces):
        raise ValueError("run needs at least one complete paired 2UI window for plots")
    (output / "config.json").write_text(json.dumps(w.config.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "waveform.npz", time=w.time, ideal=w.ideal,
                        noiseless=w.noiseless, final=w.final, symbol_indices=w.symbol_indices,
                        symbol_amplitudes=w.symbol_amplitudes, symbol_boundaries=w.symbol_boundaries,
                        valid_interval=w.valid_interval, guard_symbol_indices=w.guard_symbol_indices,
                        metadata_json=json.dumps(w.metadata), config_json=json.dumps(w.config.to_dict()))
    summary = dict(description=description, samples=len(w.time), retained_symbols=len(w.symbol_indices),
                   seed=w.config.seed, amplitude_unit=w.config.amplitude_unit, time_unit="seconds",
                   valid_interval_seconds=w.valid_interval, model_settings=w.metadata,
                   eye_trace_count=len(two.traces), plot_mode=plot_mode,
                   eye_alignment="start_k = b_k - 0.5*T + display_phase_ui*T; nominal boundary at 0.5-phase UI",
                   rendering=dict(amplitude_bins=amplitude_bins, time_bins_per_ui=time_bins_per_ui,
                                  reference_lines=reference_lines, smoothing="none", overlay_limit=500,
                                  density_line_underlay="first 500 source traces; display only; not histogram counts"),
                   versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    plot_waveform(w, output / "waveform.png")
    for length, eye in ((1, one), (2, two)):
        options = dict(amplitude_bins=amplitude_bins,
                       time_bins=None if time_bins_per_ui is None else length * time_bins_per_ui,
                       reference_lines=reference_lines)
        counts, xedges, yedges = plot_eye(eye, w, output / f"eye_{length}ui.png", plot_mode, **options)
        plot_eye(eye, w, output / f"eye_{length}ui_traces.png", "overlay", **options)
        np.savez_compressed(output / f"eye_{length}ui_counts.npz", counts=counts,
                            time_edges_ui=xedges, amplitude_edges=yedges,
                            source_start_indices=eye.source_start_indices, start_times=eye.start_times)
    if csv_interval_ui is not None:
        a, b = csv_interval_ui
        if not 0 <= a < b <= w.config.n_symbols:
            raise ValueError("CSV interval must lie within the valid interval")
        mask = (w.time >= a * w.config.symbol_period) & (w.time <= b * w.config.symbol_period)
        np.savetxt(output / "waveform_interval.csv", np.column_stack([w.time[mask], w.ideal[mask], w.noiseless[mask], w.final[mask]]),
                   delimiter=",", header="time_seconds,ideal,noiseless,final", comments="")
    return output
