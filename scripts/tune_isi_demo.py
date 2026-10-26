"""Demonstration-only bandwidth sweep; the simulation package is unchanged.

Replay the saved original retained symbols explicitly because random-mode guard
length depends on bandwidth. Explicit replay uses the generator's documented
constant endpoint guards. The comparison omits eight UIs at each end so this
guard-policy difference does not affect the displayed comparison.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from dual_tail_characterization import Config, generate_waveform, extract_eye
from dual_tail_characterization.export import save_run
from dual_tail_characterization.plotting import eye_histogram
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]


def comparison(waves, labels, output, filename):
    # Identical clock positions, axis limits, histogram edges and normalization.
    starts = np.arange(8, waves[0].config.n_symbols - 8)
    eyes = [extract_eye(w, 2, starts=starts) for w in waves]
    low = min(float(w.final.min()) for w in waves)
    high = max(float(w.final.max()) for w in waves)
    limits = (min(-.3, low - .1), max(3.3, high + .1))
    histograms = [eye_histogram(e, limits, amplitude_bins=64) for e in eyes]
    norm = LogNorm(vmin=1, vmax=max(int(h[0].max()) for h in histograms))
    fig, axes = plt.subplots(2, len(waves), figsize=(5 * len(waves), 7),
                             sharex=True, sharey=True, layout="constrained", squeeze=False)
    for column, (eye, histogram, label) in enumerate(zip(eyes, histograms, labels)):
        counts, xedges, yedges = histogram
        ax = axes[0, column]
        ax.set_facecolor("black")
        mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_equal(counts.T, 0), cmap="gray", norm=norm)
        ax.set_title(label)
        axes[1, column].plot(eye.relative_time_ui, eye.traces[:500].T, color="tab:blue", alpha=.06, lw=.6)
        axes[1, column].set_title("First 500 source traces")
        for panel in axes[:, column]:
            panel.set(xlim=(0, 2), ylim=limits, xlabel="Relative time (UI)")
            for x in (.5, 1., 1.5):
                panel.axvline(x, color="gray", lw=.5, linestyle="--")
            for level in waves[0].config.levels:
                panel.axhline(level, color="gray", lw=.5, alpha=.5)
        np.savez_compressed(output / f"{filename}_{column}_counts.npz", counts=counts,
                            time_edges_ui=xedges, amplitude_edges=yedges,
                            clock_boundary_indices=starts, source_start_indices=eye.source_start_indices)
    axes[0, 0].set_ylabel("Amplitude (arbitrary units)")
    axes[1, 0].set_ylabel("Amplitude (arbitrary units)")
    fig.colorbar(mesh, ax=axes[0, :].tolist(), label="Sample count (shared log scale)")
    fig.suptitle(f"Bandwidth-limited PAM4 · noise σ=0.035 fixed · {len(starts):,} traces/density panel\n"
                 "Identical axes, bins and color normalization; no density smoothing or line underlay")
    fig.savefig(output / f"{filename}.png", dpi=150)
    plt.close(fig)
    return dict(clock_start_first=int(starts[0]), clock_start_last=int(starts[-1]),
                density_traces=len(starts), overlay_traces=500, amplitude_limits=limits,
                amplitude_bins=64, time_bins=len(eyes[0].relative_time_ui),
                color_normalization=dict(kind="LogNorm", vmin=1, vmax=norm.vmax))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--bandwidth-ghz", type=float, default=3.)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/examples/isi_small_openings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "runs/examples/isi_demo"
    original_config = Config(**json.loads((source / "config.json").read_text()))
    original = generate_waveform(original_config)
    with np.load(source / "waveform.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(original.symbol_indices, data["symbol_indices"])
        np.testing.assert_array_equal(original.final, data["final"])
    replay = replace(original_config, symbol_mode="explicit", sequence=tuple(original.symbol_indices.tolist()))
    bandwidths = [2., 2.5, 3., 4., 6.] if args.sweep else [args.bandwidth_ghz]
    waves = [generate_waveform(replace(replay, channel_bandwidth_hz=b * 1e9)) for b in bandwidths]
    for wave in waves:
        np.testing.assert_array_equal(wave.symbol_indices, original.symbol_indices)
        np.testing.assert_allclose(wave.final - wave.noiseless, original.final - original.noiseless, atol=1e-15, rtol=0)
    labels = [f"{b:g} GHz · τ={1/(2*np.pi*b*1e9)*1e12:.2f} ps" for b in bandwidths]
    if args.sweep:
        comparison_settings = comparison(waves, labels, args.output, "bandwidth_sweep")
        report = dict(original_config=original_config.to_dict(), bandwidths_ghz=bandwidths,
                      noise_std_fixed=original_config.noise_std, seed_fixed=original_config.seed,
                      retained_sequence_verified=True, noise_samples_verified=True,
                      replay_guard_policy="Constant endpoint extension; eight UIs excluded at each end in comparison.",
                      comparison=comparison_settings,
                      selected_bandwidth_ghz=3.,
                      visual_selection="Three small openings at 3 GHz; 2.5 GHz not clearly open; 4 and 6 GHz more open than desired.")
        (args.output / "sweep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        selected = waves[0]
        save_run(selected, args.output, description=(
            "Illustrative bandwidth-limited PAM4 simulation, not a calibrated optical-link simulation. "
            "Visual small-opening selection. Original retained random symbol sequence replayed explicitly; "
            "seed and additive noise unchanged. Constant endpoint guards are used for explicit replay."))
        comparison_settings = comparison([original, selected], ["Original: 2 GHz", f"Selected: {labels[0]}"], args.output, "original_vs_selected")
        summary_path = args.output / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["demonstration_selection"] = dict(
            original_channel_bandwidth_hz=original_config.channel_bandwidth_hz,
            selected_channel_bandwidth_hz=selected.config.channel_bandwidth_hz,
            noise_std_original=original_config.noise_std, noise_std_selected=selected.config.noise_std,
            retained_sequence_sha256=hashlib.sha256(original.symbol_indices.astype("<i8").tobytes()).hexdigest(),
            comparison=comparison_settings,
            guard_note="Explicit replay holds endpoints; comparison excludes eight UIs at each end.",
            criterion="Visual demonstration only; no BER, EH/EW or compliance claim.")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(dict(bandwidths_ghz=bandwidths, noise_std=.035, sequence_verified=True,
                          comparison=comparison_settings), indent=2))


if __name__ == "__main__":
    main()
