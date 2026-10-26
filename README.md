# Continuous PAM waveform milestone

Whole-level mixture diagnostics are documented in [Internal level analysis](docs/level_analysis.md). Run `python -m dual_tail_characterization.level_benchmarks --output runs/internal_level_analysis` (with `src` on PYTHONPATH) for prescribed and sampled benchmark tables, separate from continuous waveform outputs.

Python 3.13 on Windows 11. Generate a single continuous PAM-n waveform, then extract clock-referenced, endpoint-inclusive eye windows from it. No notebook is needed. The repository was empty at implementation time.

```powershell
Set-Location 'C:\Users\henry\OneDrive\Documents\ChatGPT\dual_tail_characterization'
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\pam-waveform.exe --examples --output runs/examples
.\.venv\Scripts\pam-waveform.exe --output runs/overlay --mode overlay --csv-ui 0 16
```

Use `--config config.json` to supply any `Config` fields (unspecified fields use defaults). For example:

```json
{"levels": [-0.05, 1.25, 2.2, 2.9], "n_symbols": 6,
 "symbol_mode": "explicit", "sequence": [0, 3, 1, 2, 0, 1],
 "display_phase_ui": 0.0}
```

`n_symbols` must equal the explicit sequence length; sequences never repeat or wrap. At least one complete 2UI window is needed for the plot/export command. The waveform API also supports shorter sequences. CLI output files in an existing directory are replaced; use a new output path to retain earlier runs.

## Model and alignment

The oversampled ideal waveform changes at integer UI sample positions. A centered sampled logistic derivative shapes the entire guarded sequence in one convolution. Its scale is `tau = transition_10_90_ui / (2 ln(9))`. The half-support is rounded up to a whole sample from `tau ln(2 / tail_tolerance - 1)`, so the omitted continuous kernel mass is at most `tail_tolerance` (default 1e-10). The sampled kernel is normalized to unity DC gain. This is an idealized offline, noncausal shaping model, not an optical-device model. Discrete edge placement gives an approximately half-sample offset from the continuous logistic midpoint; requested transition duration is approximate at the selected sampling resolution. Very short transitions can be undersampled.

The optional causal first-order low-pass has `tau_channel = 1 / (2 pi bandwidth_hz)` and recurrence `y[k] = a y[k-1] + (1-a) x[k]`, where `a = exp(-dt/tau_channel)`. Bandwidth is the analog RC -3 dB parameter represented by this sampled recurrence, not an exact digital -3 dB frequency. Channel delay and memory remain in the waveform and eye diagrams.

Guard length covers logistic support plus channel settling to the configured tail tolerance, with two extra UIs. Random guards are independent draws from the configured distribution. Explicit guards hold the first/last symbol constant. Analysis excludes these guards and includes both endpoint samples: `[0, n_symbols * symbol_period]`. The right endpoint may depend on the first trailing guard symbol, intentionally; its retained symbol label is unavailable. All generated guards are exported for inspection. Gaussian sample noise is applied after filtering without clipping. SeedSequence creates separate symbol and noise generators. Reproducibility is for identical settings and runtime versions; versions are recorded.

## Eye convention and data

The default offset remains zero, but source selection is now `start_k = b_k - 0.5*T + phi*T`, where `starts` identifies nominal boundary indices k. Previously selection was `b_k + phi*T`, placing boundaries at integer relative UI. The corrected **1UI transition pattern** spans [0, 1], with a nominal transition at 0.5 UI. The **2UI eye diagram** spans [0, 2], with transitions at 0.5 and 1.5 UI and a complete opening near 1 UI. Positive offset starts the window later, moving nominal boundaries left to `0.5-phi` and `1.5-phi`; for example `--phase-ui 0.125` places them at 0.375 and 1.375 UI. Source waveform timing is unchanged. No individual trace or class is recentered, so channel crossing delays and ISI remain visible.

`extract_eye(w, window_ui, phase_ui, starts)` supports any positive integer UI length. Fractional source sample positions use linear interpolation, without circular indexing. Automatic starts keep the complete support in the valid interval; invalid explicit starts raise an error. Relative time always starts at zero. Source indices may be fractional (including zero phase with odd samples/UI). Context labels identify symbols intersecting the window, including its endpoint; `-1` means outside the retained sequence, never a measured zero.

`paired_eyes` uses identical eligible starts for 1UI and 2UI views, so each 1UI trace equals the first UI of its partner, including the shared endpoint. Unrelated endpoint histograms are not constrained. Traces are separate line artists in overlay mode (first 500 traces for readability).

Density uses every extracted numerical sample, with integer raw counts and exact bin edges saved separately. A black background and grayscale log color display use 64 amplitude bins by default, reducing sparse curves compared with the previous 240-bin display. Use `--amplitude-bins 96 --time-bins-per-ui 64` to configure binning; time bins cannot exceed the number of sample phases. No smoothing or presentation interpolation is applied to histogram counts. A labeled faint line underlay connects consecutive samples of the first 500 source traces for display only; it never changes the exported counts or edges. Deterministic noiseless density bins can still show discrete steps. Counts represent sample occurrences, not BER or tail probability. Constant transitions retain their observed populations. Both eyes share amplitude limits derived from the complete waveform and configured levels, with margins. No waveform clipping or image-pixel measurement occurs. Optional thin reference lines mark the nominal boundaries and the 2UI opening center; disable with `--no-reference-lines`.

Every run also saves continuous overlays as `eye_1ui_traces.png` and `eye_2ui_traces.png`. These use the first 500 eligible traces, without equalizing transition classes. Histogram files always retain all eligible occurrences. Rendering settings and the extraction convention are recorded in `summary.json`.

## Outputs

Each run saves `config.json`, `summary.json`, `waveform.npz`, `waveform.png`, `eye_1ui.png`, `eye_2ui.png`, and two `eye_*ui_counts.npz` files. The waveform archive includes ideal, noiseless, final, timing, symbols, guards, valid interval, configuration and alignment JSON strings, readable with `np.load(..., allow_pickle=False)`. Histogram archives include integer counts, bin edges and source starts. Optional `--csv-ui START END` saves the selected waveform interval in seconds. Large runs require memory proportional to sample and eye-window counts.

Three examples use seed 123 and 10,000 retained symbols, 128 samples/UI, 100 ps/UI, 0.20 UI shaping:

* `runs/examples/clean_pam4`: levels 0, 1, 2, 3; no noise or channel.
* `runs/examples/unequal_pam4`: levels -0.05, 1.25, 2.20, 2.90; otherwise clean.
* `runs/examples/isi_demo`: 2 GHz channel, noise sigma 0.035 arbitrary units. **Illustrative bandwidth-limited simulation, not a calibrated optical-link simulation.**

Tests check random-stream independence, level preservation, mapping/timing, transition duration and monotonicity, exact source segmentation and paired views, fractional interpolation, guard boundaries, channel memory, invalid configuration, export and raw-count conservation. No GMM, eye measurements, BER/SER, masks or optical physics are included.

## ISI demonstration with three small openings

`runs/examples/isi_small_openings` is an additional demonstration selected from a fixed-sequence bandwidth sweep at 2, 2.5, 3, 4 and 6 GHz. The original `isi_demo`, clean and unequal-level cases are preserved. The selected bandwidth is **3 GHz** (channel time constant **53.05 ps**), compared with the original 2 GHz (79.58 ps). Additive Gaussian noise stays at **sigma 0.035 arbitrary units**. Reducing deterministic channel memory opens the three eyes; noise was not reduced. Shaping remains 0.20 UI (20 ps), levels 0/1/2/3, symbol period 100 ps, 128 samples/UI (1.28 TSa/s), seed 123, display offset zero. Actual openings appear later than 1 UI because causal channel delay is retained.

The original retained random symbol sequence is replayed in explicit mode to keep all 10,000 symbols identical across bandwidths: changing bandwidth in random mode otherwise changes the guard length and retained RNG slice. The generator itself is unchanged. Explicit replay uses constant endpoint guards, so the comparison excludes eight UIs at each end. The selected full waveform export retains its documented endpoint extension. Script assertions verify identical retained symbols and identical additive noise samples (within floating-point subtraction tolerance).

From the project directory, regenerate the selected case and comparison with:

```powershell
$env:PYTHONPATH = 'src'
python scripts/tune_isi_demo.py --bandwidth-ghz 3 --output runs/examples/isi_small_openings
```

Or regenerate just the standard selected outputs from their self-contained configuration:

```powershell
python -m dual_tail_characterization.cli --config runs/examples/isi_small_openings/config.json --output runs/isi_small_openings_replay
```

To rerun the sweep:

```powershell
python scripts/tune_isi_demo.py --sweep --output runs/isi_bandwidth_sweep
```

The selected directory contains all standard waveform/eye exports plus `original_vs_selected.png` and two `original_vs_selected_*_counts.npz` archives. Comparison panels share axes, bin edges, 9,984 source windows, 500 overlay traces and one LogNorm color scale; the comparison density has no smoothing or line underlay. Counts are integer numerical histograms, separate from display normalization. `summary.json` records comparison settings and a retained-sequence hash. `runs/isi_bandwidth_sweep` contains the sweep image, raw histogram archives and selection report. This is visual illustrative selection, without a BER target, EH/EW measurement or compliance claim. Coarse histogram bins remain visible.

