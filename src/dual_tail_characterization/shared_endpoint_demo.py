"""Reproducible shared-endpoint PAM4 benchmark and validation outputs."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import norm
from .config import Config
from .waveform import generate_waveform
from .export import save_run
from .shared_endpoints import effective_lanes, lane_statistics
from .lane_benchmark import original_lane_records
from .level_analysis import Components, LevelContext, Reference, analyze_level, component_records
from .validation_workbook import RunTable, verify_workbook
import matplotlib.pyplot as plt


def _statistics(values):
    n = len(values)
    return dict(sample_count=n, sample_mean=float(np.mean(values)) if n else None,
                sample_var=float(np.var(values, ddof=1)) if n > 1 else None,
                sample_sigma=float(np.std(values, ddof=1)) if n > 1 else None,
                status="available" if n > 1 else "insufficient_samples")


def _stat_check(identifier, values, mean, variance, *, adjacent_dependence=False):
    """Six-standard-error diagnostics, conservatively allowing shared endpoints.

    Nonadjacent segments have independent endpoint fluctuations conditional on
    symbols. Adjacent observations can correlate, so use n/2 as a conservative
    effective count for mean/variance tolerances, not all along-curve samples.
    """
    n = len(values)
    if n < 2:
        return dict(check_id=identifier, sample_count=n, status="insufficient_samples",
                    mean_error=None, mean_tolerance=None, var_error=None, var_tolerance=None)
    effective_n = max(2., n / 2 if adjacent_dependence else float(n))
    mean_error = abs(float(np.mean(values)) - mean)
    var_error = abs(float(np.var(values, ddof=1)) - variance)
    mean_tolerance = 6 * np.sqrt(variance / effective_n) + 1e-12
    var_tolerance = 6 * variance * np.sqrt(2 / (effective_n - 1)) + 1e-12
    return dict(check_id=identifier, sample_count=n, effective_sample_count=effective_n,
                mean_error=mean_error, mean_tolerance=mean_tolerance,
                var_error=var_error, var_tolerance=var_tolerance,
                status="pass" if mean_error <= mean_tolerance and var_error <= var_tolerance else "fail")


def build_validation_tables(w):
    c = w.config
    if c.waveform_mode != "shared_endpoints":
        raise ValueError("shared-endpoint validation requires shared_endpoints mode")
    unit = c.amplitude_unit
    sigmas = w.metadata["endpoint_sigmas"]
    lanes = effective_lanes(c.levels, sigmas)
    original = original_lane_records()
    tables = [RunTable.from_records("original_lane_table.csv", "Original_Lane_Table", original, kind="prescribed independent-lane benchmark")]
    amplitude_units = {key: unit for key in ("mean_start", "mean_end", "sigma_start", "sigma_end")}
    tables.append(RunTable.from_records("effective_lane_table.csv", "Effective_Lane_Table", lanes, units=amplitude_units, kind="prescribed shared endpoints"))
    if c.levels == (0., 1., 2., 3.):
        comparison = [dict(start=old["start"], end=old["end"],
            original_mean_start=old["mean_start"], original_mean_end=old["mean_end"],
            original_independent_lane_sigma=old["sigma"], effective_mean_start=new["mean_start"],
            effective_mean_end=new["mean_end"], effective_sigma_start=new["sigma_start"],
            effective_sigma_end=new["sigma_end"],
            sigma_comparison_status="different parameter definitions; original sigma is not an endpoint pair")
            for old, new in zip(original, lanes)]
        tables.append(RunTable.from_records("lane_comparison.csv", "Lane_Comparison", comparison,
                     units={key: unit for key in comparison[0] if "mean" in key or "sigma" in key and key != "sigma_comparison_status"}, kind="definition comparison"))
    bit_width = max(1, (len(c.levels)-1).bit_length())
    endpoint_rows = [dict(symbol_occurrence_id=f"sym_{k:06d}", level_id=str(level), symbol_bits=format(int(level), f"0{bit_width}b"),
                          center_time_seconds=float(w.symbol_centers[k]), center_time_ui=k+.5,
                          nominal_mean=float(c.levels[level]), level_sigma=float(sigmas[level]),
                          standard_normal_z=float(w.endpoint_standard_normals[k]), endpoint_amplitude=float(value))
                     for k, (level, value) in enumerate(zip(w.symbol_indices, w.symbol_endpoint_values))]
    tables.append(RunTable.from_records("symbol_endpoints.csv", "Symbol_Endpoints", endpoint_rows,
                 units=dict(center_time_seconds="s", center_time_ui="UI", nominal_mean=unit, level_sigma=unit, endpoint_amplitude=unit), kind="realized prescribed endpoint model"))
    a, b = w.symbol_indices[:-1], w.symbol_indices[1:]
    segment_rows = [dict(segment_id=f"seg_{k:06d}", start_symbol_id=f"sym_{k:06d}", end_symbol_id=f"sym_{k+1:06d}",
                         start_level_id=str(x), end_level_id=str(y), nominal_boundary_ui=k+1., source_start_sample=k*c.samples_per_ui)
                    for k, (x, y) in enumerate(zip(a, b))]
    tables.append(RunTable.from_records("segment_labels.csv", "Segment_Labels", segment_rows,
                 units={"nominal_boundary_ui": "UI"}, kind="transmitted labels; no amplitude classification"))
    phases = (0., .5, 1.)
    theory, empirical, counts, checks, endpoint_empirical = [], [], [], [], []
    phase_values = np.stack([np.interp(np.arange(len(a))*c.samples_per_ui + u*c.samples_per_ui,
                                     np.arange(len(w.final)), w.final) for u in phases], axis=1)
    for lane in lanes:
        start, end = lane["start"], lane["end"]
        mask = (a == start) & (b == end)
        counts.append(dict(transition_id=f"{start}->{end}", start_level_id=str(start), end_level_id=str(end),
                           segment_count=int(mask.sum()), status="populated" if mask.any() else "not_observed"))
        for p, u in enumerate(phases):
            mean, variance, sigma = lane_statistics(lane["mean_start"], lane["mean_end"], lane["sigma_start"], lane["sigma_end"], u, w.metadata["effective_kappa"])
            key = dict(transition_id=f"{start}->{end}", start_level_id=str(start), end_level_id=str(end), phase_u=u)
            theory.append(dict(**key, mean=float(mean), var=float(variance), sigma=float(sigma),
                               reference_status="analytic_pre_impairment_independent_endpoints"))
            selected = phase_values[mask, p]
            empirical.append(dict(**key, **_statistics(selected), variance_estimator="ddof=1; one observation per segment at this phase"))
            checks.append(_stat_check(f"lane_{start}_{end}_u_{u:g}", selected, float(mean), float(variance), adjacent_dependence=True))
    for level in range(len(c.levels)):
        selected = w.symbol_endpoint_values[w.symbol_indices == level]
        endpoint_empirical.append(dict(level_id=str(level), **_statistics(selected)))
        checks.append(_stat_check(f"endpoint_level_{level}", selected, c.levels[level], sigmas[level]**2))
    for identifier, passed in (
        ("all_transition_classes_populated", all(row["segment_count"] > 0 for row in counts)),
        ("shared_endpoints_exact", np.array_equal(w.final[::c.samples_per_ui], w.symbol_endpoint_values)),
        ("global_time_strictly_increasing", bool(np.all(np.diff(w.time) > 0))),
    ):
        checks.append(dict(check_id=identifier, status="pass" if passed else "fail"))
    checks.append(dict(check_id="fitted_or_eye_boundary_metrics", status="not_implemented", mean_error=None,
                       reason="No fitter or eye-characterization module is fabricated"))
    tables.extend([
        RunTable.from_records("transition_counts.csv", "Transition_Counts", counts),
        RunTable.from_records("lane_theory.csv", "Lane_Theory", theory, units=dict(phase_u="UI within segment", mean=unit, var=f"{unit}**2", sigma=unit), kind="theoretical pre-impairment"),
        RunTable.from_records("lane_empirical.csv", "Lane_Empirical", empirical, units=dict(phase_u="UI within segment", sample_mean=unit, sample_var=f"{unit}**2", sample_sigma=unit)),
        RunTable.from_records("endpoint_empirical.csv", "Endpoint_Empirical", endpoint_empirical, units=dict(sample_mean=unit, sample_var=f"{unit}**2", sample_sigma=unit)),
        RunTable.from_records("validation_summary.csv", "Validation_Summary", checks),
    ])
    level_rows, components = [], []
    for level in range(len(c.levels)):
        context = LevelContext("shared_endpoint_prescribed", str(level), unit, .5, (.5, .5))
        parameters = Components([1.], [c.levels[level]], [sigmas[level]])
        level_rows.append(analyze_level(parameters, context, Reference(c.levels[level], reference_kind="configured_nominal")))
        components.extend(component_records(parameters, context))
    tables.extend([RunTable.from_records("level_analysis.csv", "Level_Analysis", level_rows, kind="prescribed whole-level endpoint metrics; not fitted"),
                   RunTable.from_records("component_parameters.csv", "Component_Parameters", components, kind="prescribed endpoint parameters")])
    return tables, phase_values, checks


def plot_validation_histograms(w, phase_values, output):
    unit = w.config.amplitude_unit
    low, high = float(w.final.min()), float(w.final.max())
    edges = np.linspace(low - .05*(high-low), high + .05*(high-low), 121)
    centers = (edges[:-1] + edges[1:]) / 2
    endpoint_counts = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for level, ax in enumerate(axes.flat):
        selected = w.symbol_endpoint_values[w.symbol_indices == level]
        counts, _ = np.histogram(selected, edges)
        endpoint_counts.append(counts)
        ax.stairs(counts, edges, label="Numerical counts")
        sigma = w.metadata["endpoint_sigmas"][level]
        if sigma:
            ax.plot(centers, norm.pdf(centers, w.config.levels[level], sigma) * len(selected) * np.diff(edges), label="Analytic Gaussian")
        ax.set(title=f"Endpoint level {level} · n={len(selected):,}", xlabel=f"Amplitude ({unit})", ylabel="Sample count", xlim=(edges[0], edges[-1]))
        ax.legend(fontsize=8)
    fig.savefig(output / "endpoint_level_histograms.png", dpi=150)
    plt.close(fig)
    a, b = w.symbol_indices[:-1], w.symbol_indices[1:]
    all_counts = np.stack([[np.histogram(phase_values[(a == start) & (b == end), p], edges)[0]
                            for p in range(3)] for start in range(4) for end in range(4)])
    selected_pairs = [(0, 0), (0, 3), (3, 0), (3, 3)]
    fig, axes = plt.subplots(4, 3, figsize=(12, 10), layout="constrained")
    for r, (start, end) in enumerate(selected_pairs):
        mask = (a == start) & (b == end)
        for p, u in enumerate((0., .5, 1.)):
            ax = axes[r, p]
            ax.stairs(all_counts[start*4+end, p], edges)
            sigmas = w.metadata["endpoint_sigmas"]
            mean, _, sigma = lane_statistics(w.config.levels[start], w.config.levels[end], sigmas[start], sigmas[end], u, w.metadata["effective_kappa"])
            if sigma:
                ax.plot(centers, norm.pdf(centers, mean, sigma) * mask.sum() * np.diff(edges), lw=1)
            ax.set(title=f"{start}→{end} · u={u:g} · n={mask.sum():,}", xlabel=f"Amplitude ({unit})", ylabel="Sample count", xlim=(edges[0], edges[-1]))
    fig.suptitle("Transition-conditioned counts and analytic references · actual transmitted labels")
    fig.savefig(output / "transition_conditioned_histograms.png", dpi=150)
    plt.close(fig)
    np.savez_compressed(output / "validation_histogram_counts.npz", endpoint_counts=np.asarray(endpoint_counts),
                        transition_phase_counts=all_counts, amplitude_edges=edges, phases_u=[0., .5, 1.],
                        ordered_pairs=np.array([(a,b) for a in range(4) for b in range(4)]))


def run_demo(output, *, n_symbols=32_000, seed=123):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = Config(waveform_mode="shared_endpoints", endpoint_sigmas=(.2, .1, .1, .2),
                    n_symbols=n_symbols, seed=seed)
    waveform = generate_waveform(config)
    tables, phase_values, checks = build_validation_tables(waveform)
    waveform.metadata["validation_status"] = "pass" if all(r["status"] in ("pass", "not_implemented") for r in checks) else "incomplete_or_failed"
    waveform.metadata["validation_tolerances"] = "6 standard errors; lane n_eff=n/2 conservatively accounts for adjacent endpoint sharing"
    plot_validation_histograms(waveform, phase_values, output)
    save_run(waveform, output, description="Shared-symbol-endpoint PAM4; original independent-lane offsets removed deliberately; no channel, timing jitter or additive sample noise.", extra_tables=tables)
    # Reopen the actual workbook and compare every table cell against its source.
    import json
    from .validation_workbook import configuration_table
    summary = json.loads((output / "summary.json").read_text())
    all_tables = [configuration_table(config.to_dict()), RunTable.from_records("run_summary.csv", "Run_Summary", [summary], kind="metadata"), *tables]
    mapping = json.loads((output / "workbook_mapping.json").read_text())
    verified = verify_workbook(all_tables, output, mapping)
    (output / "workbook_verification.json").write_text(json.dumps(dict(status="pass", verified_cells=verified,
        csv_tables=len(all_tables), worksheets=len(mapping)+1), indent=2), encoding="utf-8")
    write_report(output)
    return output


def write_report(output):
    """Human-readable index to the numerical validation results of this run."""
    import csv
    import json
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    config = json.loads((output / "config.json").read_text())
    verified = json.loads((output / "workbook_verification.json").read_text())
    with (output / "transition_counts.csv").open(newline="", encoding="utf-8") as file:
        populations = [int(row["segment_count"]) for row in csv.DictReader(file)]
    model = summary["model_settings"]
    report = f"""# Shared-symbol-endpoint PAM4 validation report

Run status: **{model['validation_status']}**. The {config['n_symbols']:,} symbol occurrences populate all 16 ordered transitions, with {min(populations):,}–{max(populations):,} segments per class.

## Inputs and effective model

* Nominal symbol means: {config['levels']} {config['amplitude_unit']}.
* Endpoint amplitude sigmas: {config['endpoint_sigmas']} {config['amplitude_unit']}.
* Symbol period: {config['symbol_period']*1e12:g} ps; {config['samples_per_ui']} samples/UI; seed {config['seed']}.
* Solved kappa: {model['effective_kappa']:.14g}; normalized logistic 10–90 duration: {model['effective_transition_10_90_ui']:.14g} UI.
* No channel, additive sample noise or timing jitter. Centers are at half-integer physical UI; display transitions are at 0.5 and 1.5 UI.

The original independent-lane table is preserved as [original_lane_table.csv](original_lane_table.csv). It contains exactly 16 unique pairs, including the corrected 2→3 row. Its single lane sigma is not a pair of endpoint sigmas. [effective_lane_table.csv](effective_lane_table.csv) derives means and sigmas from symbol properties, deliberately removing original transition-dependent offsets. [lane_comparison.csv](lane_comparison.csv) shows both definitions explicitly.

## Validation results

[validation_summary.csv](validation_summary.csv) records endpoint and lane checks. There is one observation per segment per selected phase, with actual transition labels. Six-standard-error tolerances conservatively use n/2 for lane populations to allow adjacent endpoint sharing. Same-symbol segments retain independent endpoint fluctuations and have half their endpoint variance at the midpoint. No fitted or eye-boundary measurements are fabricated.

All {verified['csv_tables']} CSV tables appear in [validation_results.xlsx](validation_results.xlsx), with {verified['worksheets']} worksheets including README. Reopening with openpyxl verified {verified['verified_cells']:,} cells against their typed source tables. The workbook README maps CSV files to sheets and defines units, nulls and quantity origins. [workbook_mapping.json](workbook_mapping.json) and [workbook_verification.json](workbook_verification.json) provide machine-readable checks.

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
python -m dual_tail_characterization.shared_endpoint_demo --output runs/shared_endpoints_pam4 --n-symbols {config['n_symbols']} --seed {config['seed']}
```

Install with `python -m pip install -e '.[test]'` in a Python 3.13 environment if needed. No workbook formulas are used to simulate physics. Native Excel interaction is not required for reproduction; file contents are checked with openpyxl.

## Implementation files

New modules: `shared_endpoints.py`, `lane_benchmark.py`, `shared_endpoint_demo.py`, and `validation_workbook.py` under `src/dual_tail_characterization`. The independent-lane source is `benchmarks/customized_pam4_independent_lanes.csv`.

Updated modules: `config.py`, `waveform.py`, `eyes.py`, `plotting.py`, `export.py`, `cli.py`, `level_analysis_export.py`, and `level_benchmarks.py`. Dependency metadata now includes openpyxl. Tests add `test_shared_endpoints.py` and extend waveform and level-analysis export coverage. Documentation is in `docs/shared_endpoints.md` and the README. Existing example output files were not rewritten.

## Limitations

Amplitude continuity does not imply derivative continuity. Adjacent segments are correlated. Analytic references assume independent endpoint fluctuations before impairments; the initial shared mode rejects channel filtering and additive noise. Histogram binning remains visible. Excel native numeric precision applies; NPZ remains authoritative for full floating-point data. Existing continuous-filter examples are preserved.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/shared_endpoints_pam4"))
    parser.add_argument("--n-symbols", type=int, default=32_000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(run_demo(args.output, n_symbols=args.n_symbols, seed=args.seed))


if __name__ == "__main__":
    main()
