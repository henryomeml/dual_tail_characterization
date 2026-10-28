# Shared-symbol-endpoint waveform validation

This is a separate `waveform_mode="shared_endpoints"`. The existing `continuous_filter` mode, original examples, and whole-level variance analysis remain available. The initial shared-endpoint mode disables channel filtering and additive sample noise, and implements no timing jitter.

## Preserved independent-lane benchmark

The authoritative 16 ordered PAM4 pairs are preserved in `benchmarks/customized_pam4_independent_lanes.csv` and exported as `original_lane_table.csv`. The corrected 2→3 row is `(2,3,2,2.97,0.1)`, once only. The unusual 2→2 mean-end value 2.97 remains as requested. No contradictory row is used.

That table's single sigma is an **independent-lane parameter**. It is not copied into both endpoint sigmas and does not describe the variance along a new shared-endpoint segment.

The demonstration defines nominal per-symbol means `[0,1,2,3]` with amplitude sigmas `[0.2,0.1,0.1,0.2]`. Its effective a→b properties are:

```text
mean_start  = nominal_mean[a]
mean_end    = nominal_mean[b]
sigma_start = level_sigma[a]
sigma_end   = level_sigma[b]
```

Transition-dependent mean offsets from the original benchmark are deliberately removed. Both definitions and an explicit comparison are exported, without overwriting the benchmark. For example, original 0→3 has means 0.09 and 2.91, and independent-lane sigma 0.2. The effective shared segment has means 0 and 3, endpoint sigmas 0.2 and 0.2, and midpoint sigma `sqrt((0.2²+0.2²)/4)`, approximately 0.141421.

## Endpoint and segment construction

Each symbol occurrence k gets one saved amplitude `V_k = A[s_k] + sigma[s_k]*Z_k`. Separate `SeedSequence(seed).spawn(3)` streams select symbols, endpoint standard-normal deviations, and reserved additive noise. Endpoint sigma is amplitude variation, not timing RJ. Altering endpoint sigmas leaves the selected symbols and saved Z values unchanged.

For phase u in [0,1], `g(u)=expit(kappa*(u-0.5))` and `L(u)=(g(u)-g(0))/(g(1)-g(0))`. An algebraically equivalent tanh ratio avoids cancellation for small kappa. The near-linear limit below kappa=1e-6 has errors below approximately 1e-14. Exact endpoint weights are 0 and 1.

`Y_k(u) = (1-L(u))*V_k + L(u)*V_(k+1)`. The equivalent difference form used in code keeps identical endpoints exactly constant. Each shared center sample occurs once in the global array; the next segment uses that same saved amplitude. N symbols produce N-1 segments and `(N-1)*samples_per_ui+1` samples. Explicit sequences are never repeated, wrapped or extended. No guard data is necessary because there is no convolution in this mode.

Amplitude continuity is guaranteed. Derivatives generally differ at joins. Two neighboring segments are correlated because they share an endpoint. Same-symbol lanes have a horizontal ensemble mean, but individual segments between independently fluctuating endpoints are not forced flat.

## Duration and phase convention

Kappa must be finite and positive. If `shared_kappa` is absent, the configured `transition_10_90_ui` (default 0.20) is solved for the normalized curve using:

```text
duration_10_90 = (4/kappa)*atanh(0.8*tanh(kappa/4))
```

This normalized duration must lie strictly between zero and 0.8 UI. An explicitly supplied kappa takes precedence, and the actual normalized duration is recorded in metadata. The usual unnormalized formula is not asserted to remain exact. Very steep curves can be undersampled at a low samples/UI setting.

Symbol centers are at `(k+0.5)*T`; nominal symbol boundaries are at integer UI. The valid interval is the first through last symbol center, `[0.5*T,(N-0.5)*T]`. NPZ explicitly distinguishes `symbol_centers` from `symbol_boundaries`. `symbol_amplitudes` remain the nominal mapped means; `symbol_endpoint_values` are realized V values.

The existing eye API still uses `start_k = b_k - 0.5*T + display_phase_ui*T`. In shared mode, boundary index k identifies the segment between symbols k-1 and k. With zero phase, the first 1UI trace starts at 0.5*T and spans the first two centers. Its transition midpoint is displayed at 0.5 UI. The 2UI trace spans three centers, with nominal transitions at 0.5 and 1.5 UI and its central saved endpoint at 1 UI. Positive display phase moves boundaries left on both displays. Fractional source positions use linear interpolation on the same waveform; only complete windows are eligible. Shared-mode context labels identify endpoint symbols, while continuous-filter context remains unchanged.

## Theory and empirical validation

Conditioned on the transmitted a→b labels:

```text
mean_ab(u) = (1-L)*A[a] + L*A[b]
var_ab(u)  = (1-L)^2*sigma[a]^2 + L^2*sigma[b]^2
sigma_ab(u) = sqrt(var_ab(u))
```

At u=0.5, the mean is `(A[a]+A[b])/2` and variance is `(sigma[a]^2+sigma[b]^2)/4`. For a→a, variance varies along the curve, attaining half the endpoint variance at the midpoint. No fixed lane sigma is substituted.

Validation uses actual transmitted symbols, not classification by observed amplitude. Each lane's sample mean/variance/standard deviation uses one observation per segment at each of u=0, 0.5, and 1. Classes are never pooled and called a component sigma. Raw phase histograms for all 16 classes and three phases are saved. Adjacent phases/segments share data and are not treated as mutually independent.

Statistical diagnostic tolerances use six standard errors plus 1e-12 amplitude/variance units. For lane checks, effective count n/2 conservatively allows correlation from neighboring segments; endpoint checks use the actual independent endpoint count. Sample variances use ddof=1. Insufficient populations are marked unavailable with null values. The initial demo has no impairments, so analytic references can be compared directly to samples. They remain labeled pre-impairment references for future extensions. No post-filter recovery claim, GMM fit, EH/EW or BER output is fabricated.

## Reproduce

```powershell
Set-Location 'C:\Users\henry\OneDrive\Documents\ChatGPT\dual_tail_characterization'
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m dual_tail_characterization.shared_endpoint_demo --output runs/shared_endpoints_pam4 --n-symbols 32000 --seed 123
```

Using existing installed dependencies:

```powershell
$env:PYTHONPATH = 'src'
python -m dual_tail_characterization.shared_endpoint_demo --output runs/shared_endpoints_pam4 --n-symbols 32000 --seed 123
```

The main CLI also provides `pam-waveform --shared-endpoint-demo --output runs/shared_endpoints_pam4` for the fixed demonstration. For arbitrary PAM-n inputs, use a JSON configuration and the existing CLI:

```json
{"waveform_mode":"shared_endpoints", "levels":[-0.2,0.75,2.9],
 "endpoint_sigmas":[0.3,0.0,0.1], "n_symbols":10000,
 "transition_10_90_ui":0.2, "seed":123}
```

```powershell
python -m dual_tail_characterization.cli --config shared_config.json --output runs/shared_custom
```

An absent `endpoint_sigmas` means zero fluctuation for every configured level. Channel and sample noise settings other than disabled are rejected in this first version. The three RNG streams are reserved consistently for later separately labeled impairments.

## Outputs

`runs/shared_endpoints_pam4` contains configuration/summary JSON, complete `waveform.npz`, the waveform plot, 1UI/2UI density and trace plots, eye raw count archives, `endpoint_level_histograms.png`, `transition_conditioned_histograms.png`, and `validation_histogram_counts.npz` (integer counts and exact bin edges).

CSV tables cover original/effective/comparison lane definitions, per-occurrence endpoints and segment labels, transition counts, lane theory, conditional empirical statistics, endpoint statistics, validation results, prescribed whole-level endpoint analysis, components, configuration and run summary. The empirical endpoint statistics are separate from prescribed whole-level metrics. Existing density rendering settings and raw waveform values are preserved without clipping.

## One workbook per run

`validation_results.xlsx` contains every CSV table generated by the run, using the same typed table objects, values and row order. `README` maps each CSV to worksheet parts, with source row/column ranges, units, null conventions and distinctions between prescribed, theoretical, empirical and fitted quantities. `workbook_mapping.json` exposes the same map programmatically. `workbook_verification.json` reports a complete openpyxl read-back comparison for the demonstration.

Rows and columns exceeding Excel's limits are split into uniquely named sheets (at most 31 characters), with repeated headers and a complete mapping. Header rows are frozen, filters enabled, and widths/number formats applied. Numeric values are numeric cells without pre-rounding. Text IDs and leading-zero bit strings remain text; text beginning with '=' is written as literal text. Missing numeric values remain blank with status fields. Excel's native numeric precision still applies; NPZ is authoritative for complete floating-point waveform data. Strings exceeding Excel's 32767-character cell limit and numeric integers exceeding 15 digits fail explicitly rather than truncate; use separate source fields or text identifiers.

No full waveform is placed in Excel unless `--csv-ui` explicitly requests an interval table. Per-symbol endpoints are a separate requested validation table. The normal waveform exporter and level benchmark runner also now consolidate their produced CSV tables into a workbook. Previously saved runs are not rewritten automatically. No workbook formulas simulate physics; worksheets are validation snapshots of numerical outputs.
