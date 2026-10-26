# Internal whole-level amplitude analysis

`dual_tail_characterization.level_analysis` accepts known component parameters directly. It has no dependency on the waveform generator, eye plotting, GMM fitting, tail truncation, or boundary optimization. DJ/RJ names describe model-defined amplitude variance, not timing jitter and not an attribution to independent physical noise sources.

## API

```python
from dual_tail_characterization.level_analysis import (
    Components, LevelContext, Reference, analyze_level, component_records,
    estimate_lane_reference,
)
from dual_tail_characterization.level_analysis_export import export_table

parameters = Components.from_offsets(
    nominal_amplitude=0., weights=[.5, .5], offsets=[-.3, .3],
    sigmas=[.1, .1], require_mean_preservation=True,
)
context = LevelContext(
    dataset_id="controlled_bottom_split", level_id=0,
    amplitude_unit="V", sampling_phase_ui=.5,
    sampling_window_ui=(.5, .5), parameter_origin="prescribed_simulation",
    channel_id=None, wavelength_nm=None,
)
reference = Reference(mu_ref=0., reference_kind="configured_nominal")
row = analyze_level(parameters, context, reference, direction_tolerance=1e-12)
export_table([row], "runs/my_analysis/internal_level_analysis")
export_table(component_records(parameters, context), "runs/my_analysis/components")
```

Alternatively construct `Components(weights, means, sigmas)` directly. Vectors must be nonempty, finite, one dimensional and equal length. Weights must be nonnegative and sum to one within `weight_sum_tolerance` (default absolute tolerance 1e-12). Accepted weights are used as supplied; no normalization occurs. The sum and tolerance are recorded. Zero-weight components are retained in the parameter table and ignored in the sums. All standard deviations must be nonnegative; zero is allowed, including deterministic prescribed cases. A future fitter may impose its own positive floor. Derived overflow fails clearly rather than exporting infinities.

`from_offsets` defines `mu_k = nominal_amplitude + offset_k`. Setting `require_mean_preservation=True` requires `abs(sum(w_k * offset_k)) <= mean_tolerance` (default 1e-12 amplitude units). This is explicit, not an automatic offset correction.

`parameter_origin` is `prescribed_simulation` or `fitted_measurement`. Metadata carries dataset/level identifiers, amplitude unit, phase and window, optional channel identifier and wavelength in nm. Sampling phase/window are in UI after a nominal symbol boundary; they describe where a population is defined or selected, not a timing-jitter output. A point population may use `(phase, phase)`; an absent window denotes a point at the specified phase. No WDM generation, wavelength aggregation or crosstalk attribution is performed.

## Equations and units

For each complete level population:

```text
mu_g     = sum(w_k * mu_k)
DJ_var   = sum(w_k * (mu_k - mu_g)**2)
RJ_var   = sum(w_k * sigma_k**2)
TJ_var   = DJ_var + RJ_var
sigma_DJ = sqrt(DJ_var)
sigma_RJ = sqrt(RJ_var)
sigma_TJ = sqrt(TJ_var)
```

Variance units are `(amplitude_unit)**2`; RMS deviations have amplitude units. The mixture random variance is weighted. No `sqrt(sum(sigma_k**2))` or independent-source interpretation is used.

For a valid reference:

```text
delta_g = mu_g - mu_ref
mu_abs  = mu_ref + sign(delta_g) * sigma_DJ
```

`mu_abs_name` is **Reference-based absolute DJ mean**. This descriptive marker is not generally a mode, quantile, decision threshold, or inward-tail boundary. Every row retains `sigma_DJ` alongside it. Exact `delta_g == 0` uses sign zero, returning `mu_ref` even with positive `DJ_var`. The direction status is `exact_zero`, `positive`, `negative`, `near_zero_positive`, `near_zero_negative`, or `unavailable`. `direction_tolerance` (default 1e-12 amplitude units) only flags nonzero deltas at or below that magnitude. It never changes the sign used in the formula. Thus the marker may be directionally unstable near zero; this is reported rather than silently corrected.

## References and provenance

* `configured_nominal` explicitly supplies the configured simulation level. It is not waveform-estimated.
* `steady_state_lane` requires a selection rule, phase/window, positive selected sample count and estimation method. The caller must select and justify the population; no largest-component heuristic is used.
* `user_supplied` supports an explicitly supplied finite reference without claiming either of those origins.

For a lane already selected by the caller:

```python
reference = estimate_lane_reference(
    selected_lane_samples,
    selection_rule="known symbol 1, run length >= 8; phase-window samples only",
    sampling_phase_ui=.5, sampling_window_ui=(.45, .55),
)
```

The helper computes an arithmetic mean and sample standard deviation (`ddof=1`) and requires at least two finite samples by default. It records the rule and sample count. It does not verify the rule against a waveform or select samples itself. Invalid or insufficient populations yield an invalid Reference with a reason.

An absent reference leaves `reference_status="missing"`; a nonfinite mean, invalid optional sigma, incomplete lane provenance or explicit `validity=False` leaves `reference_status="invalid"`. Both preserve whole-level metrics and return null `mu_ref`, `sigma_ref`, `delta_g`, and `mu_abs`, with a reason and unavailable direction. Reference sigma is optional provenance only and does not alter mixture metrics. No absolute-RJ quantity is computed.

JSON exports use strict nulls, never NaN or zero placeholders. CSV uses empty cells for unavailable values and JSON text for nested provenance/window fields. Invalid nonfinite reference metadata is serialized as null, with its invalidity recorded. Component parameters are a separate long-form table.

## Controlled benchmarks and reproduction

```powershell
Set-Location 'C:\Users\henry\OneDrive\Documents\ChatGPT\dual_tail_characterization'
$env:PYTHONPATH = 'src'
python -m pytest -q
python -m dual_tail_characterization.level_benchmarks --output runs/internal_level_analysis --sigma 0.1 --splits 0 0.15 0.3 --sample-count 50000 --seed 123
```

The three scenarios preserve nominal PAM4 means. Each has middle levels 1 and 2 with a single sigma=0.1 component, top level 3 with sigma=0.2, and bottom level 0 with equal-weight means `[-d, +d]` and component sigma=0.1. All use configured nominal references and phase 0.5 UI, point window `[0.5, 0.5]`. At d=0 the two bottom components coincide; they remain two prescribed labels, not an inferred component count.

| Bottom split d | mu_g | DJ_var | RJ_var | TJ_var | sigma_DJ | mu_abs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.01 | 0.01 | 0 | 0 |
| 0.15 | 0 | 0.0225 | 0.01 | 0.0325 | 0.15 | 0 |
| 0.30 | 0 | 0.09 | 0.01 | 0.10 | 0.30 | 0 |

Top-level `RJ_var=0.04`, four times either middle level's 0.01. Increasing the bottom split preserves component sigma but increases the total level standard deviation.

Outputs, under a separate `internal_level_analysis` group:

* `prescribed_level_analysis.json/.csv`: exact parameter-based whole-level metrics and reference markers.
* `component_parameters.json/.csv`: component weights, means, sigmas, nominal amplitudes and offsets.
* `empirical_statistics.json/.csv`: sampled level means, sample variances and sample standard deviations.
* `empirical_component_statistics.json/.csv`: counts and statistics using the saved prescribed labels.
* `sample_populations.npz`: numerical samples and component labels for every scenario/level.
* `fitted_level_analysis.json/.csv`: empty fitted records; `manifest.json` explicitly states that fitting is unavailable and was not performed.
* `manifest.json`: complete prescribed configurations, tolerances, random-stream rule, seed, units, phase/window and versions.

Labels are selected independently from prescribed categorical probabilities. Offsets therefore represent a **between-component statistical structure**, not a claim about deterministic physical sources. Labels are saved for every draw. Separate SeedSequence streams drive labels and within-component Gaussian samples. Empirical records are direct population draws, not continuous waveform realizations; filtering and transition overlap may change measured phase-conditional distributions. No recovery of prescribed parameters after filtering is asserted.

## Separation from boundary measurements

No dual-tail calculation is changed or added. Future EH must continue to use full conditional-distribution inverse CDFs: `a_lower_i = inverse_CDF_i(1-alpha_lower_i)`, `a_upper_i = inverse_CDF_(i+1)(alpha_upper_i)`, and `EH_i = a_upper_i-a_lower_i`. Neither `mu_abs`, `mu_g +/- z*sigma_TJ`, nor a dominant component is a replacement. No BER from DJ/RJ or Q, no side-specific/truncated DJ/RJ tables, and no GMM fitter are introduced.
