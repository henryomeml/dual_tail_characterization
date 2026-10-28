# Shared-symbol-endpoint PAM4 validation report

Run status: **pass**. The 32,000 symbol occurrences populate all 16 ordered transitions, with 1,871–2,048 segments per class.

## Inputs and effective model

* Nominal symbol means: [0.0, 1.0, 2.0, 3.0] arbitrary units.
* Endpoint amplitude sigmas: [0.2, 0.1, 0.1, 0.2] arbitrary units.
* Symbol period: 100 ps; 128 samples/UI; seed 123.
* Solved kappa: 21.970739414303; normalized logistic 10–90 duration: 0.2 UI.
* No channel, additive sample noise or timing jitter. Centers are at half-integer physical UI; display transitions are at 0.5 and 1.5 UI.

The original independent-lane table is preserved as [original_lane_table.csv](original_lane_table.csv). It contains exactly 16 unique pairs, including the corrected 2→3 row. Its single lane sigma is not a pair of endpoint sigmas. [effective_lane_table.csv](effective_lane_table.csv) derives means and sigmas from symbol properties, deliberately removing original transition-dependent offsets. [lane_comparison.csv](lane_comparison.csv) shows both definitions explicitly.

## Validation results

[validation_summary.csv](validation_summary.csv) records endpoint and lane checks. There is one observation per segment per selected phase, with actual transition labels. Six-standard-error tolerances conservatively use n/2 for lane populations to allow adjacent endpoint sharing. Same-symbol segments retain independent endpoint fluctuations and have half their endpoint variance at the midpoint. No fitted or eye-boundary measurements are fabricated.

All 14 CSV tables appear in [validation_results.xlsx](validation_results.xlsx), with 15 worksheets including README. Reopening with openpyxl verified 514,214 cells against their typed source tables. The workbook README maps CSV files to sheets and defines units, nulls and quantity origins. [workbook_mapping.json](workbook_mapping.json) and [workbook_verification.json](workbook_verification.json) provide machine-readable checks.

## Numerical and visual outputs

* [waveform.npz](waveform.npz): complete waveform, transmitted symbols, saved endpoints/Z values, centers, boundaries, valid interval and model metadata.
* [waveform.png](waveform.png), [eye_1ui.png](eye_1ui.png), [eye_2ui.png](eye_2ui.png), and separate `eye_*ui_traces.png` overlays.
* [endpoint_level_histograms.png](endpoint_level_histograms.png) and [transition_conditioned_histograms.png](transition_conditioned_histograms.png).
* `eye_*ui_counts.npz` and [validation_histogram_counts.npz](validation_histogram_counts.npz): raw numerical counts and exact bin edges, independent of display normalization.
* [lane_theory.csv](lane_theory.csv), [lane_empirical.csv](lane_empirical.csv), [endpoint_empirical.csv](endpoint_empirical.csv), [symbol_endpoints.csv](symbol_endpoints.csv), and [segment_labels.csv](segment_labels.csv).

## Reproduction

From the project directory in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m pytest -q
python -m dual_tail_characterization.shared_endpoint_demo --output runs/shared_endpoints_pam4 --n-symbols 32000 --seed 123
```

Install with `python -m pip install -e '.[test]'` in a Python 3.13 environment if needed. No workbook formulas are used to simulate physics. Native Excel interaction is not required for reproduction; file contents are checked with openpyxl.

## Limitations

Amplitude continuity does not imply derivative continuity. Adjacent segments are correlated. Analytic references assume independent endpoint fluctuations before impairments; the initial shared mode rejects channel filtering and additive noise. Histogram binning remains visible. Excel native numeric precision applies; NPZ remains authoritative for full floating-point data. Existing continuous-filter examples are preserved.
