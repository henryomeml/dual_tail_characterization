import argparse
import json
from dataclasses import replace
from pathlib import Path
from .config import Config
from .waveform import generate_waveform
from .export import save_run


def main():
    parser = argparse.ArgumentParser(description="Continuous PAM waveform and paired eye views")
    parser.add_argument("--config", type=Path, help="JSON Config fields")
    parser.add_argument("--output", type=Path, default=Path("runs/default"))
    parser.add_argument("--examples", action="store_true", help="Generate three documented demonstration cases")
    parser.add_argument("--mode", choices=["density", "overlay"], default="density")
    parser.add_argument("--phase-ui", type=float, help="Display offset (default Config: 0): start=b_k-0.5T+phase*T; positive moves boundaries left")
    parser.add_argument("--amplitude-bins", type=int, default=64)
    parser.add_argument("--time-bins-per-ui", type=int, help="Density time bins/UI; at most samples/UI (default: one bin/sample phase)")
    parser.add_argument("--no-reference-lines", action="store_true")
    parser.add_argument("--csv-ui", type=float, nargs=2, metavar=("START", "END"))
    args = parser.parse_args()
    rendering = dict(amplitude_bins=args.amplitude_bins, time_bins_per_ui=args.time_bins_per_ui,
                     reference_lines=not args.no_reference_lines)
    if args.examples and args.config:
        parser.error("--examples and --config are mutually exclusive")
    if args.examples:
        cases = [("clean_pam4", Config(), "Clean PAM4: 0.20 UI logistic transition; no channel or noise."),
                 ("unequal_pam4", Config(levels=(-.05, 1.25, 2.20, 2.90)), "Unequal-level clean PAM4."),
                 ("isi_demo", Config(channel_bandwidth_hz=2e9, noise_std=.035),
                  "Illustrative bandwidth-limited simulation, not a calibrated optical-link simulation. 2 GHz first-order channel; Gaussian noise sigma 0.035 arbitrary units.")]
        for name, config, description in cases:
            if args.phase_ui is not None:
                config = replace(config, display_phase_ui=args.phase_ui)
            print(save_run(generate_waveform(config), args.output / name, args.mode, description, args.csv_ui, **rendering))
    else:
        config = Config(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else Config()
        if args.phase_ui is not None:
            config = replace(config, display_phase_ui=args.phase_ui)
        print(save_run(generate_waveform(config), args.output, args.mode, csv_interval_ui=args.csv_ui, **rendering))


if __name__ == "__main__":
    main()
