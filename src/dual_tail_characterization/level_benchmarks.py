"""Prescribed PAM4 population benchmarks, separate from continuous waveforms."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import numpy as np
from .level_analysis import Components, LevelContext, Reference, analyze_level, component_records
from .level_analysis_export import export_table
from .validation_workbook import configuration_table, write_workbook


def run_benchmarks(output, *, sigma=.1, splits=(0., .15, .3), sample_count=50_000, seed=123):
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if not splits or any(not np.isfinite(d) or d < 0 for d in splits):
        raise ValueError("splits must contain finite nonnegative offsets")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("sample_count must be an integer >= 2")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    label_rng, noise_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    theoretical, parameters, empirical, empirical_components = [], [], [], []
    populations = {}
    configurations = []
    for scenario, d in enumerate(splits):
        dataset_id = f"prescribed_split_{scenario}"
        for level in range(4):
            weights, offsets, sigmas = ([.5, .5], [-d, d], [sigma, sigma]) if level == 0 else ([1.], [0.], [sigma * (2 if level == 3 else 1)])
            components = Components.from_offsets(float(level), weights, offsets, sigmas, require_mean_preservation=True)
            context = LevelContext(dataset_id, level, "arbitrary units", .5, (.5, .5))
            reference = Reference(float(level), reference_kind="configured_nominal")
            row = analyze_level(components, context, reference)
            row.update(record_type="prescribed_theoretical", split_d=d,
                       component_assignment="independent random labels; between-component statistical structure, not a deterministic physical source")
            theoretical.append(row)
            parameters.extend(component_records(components, context))
            labels = label_rng.choice(len(weights), sample_count, p=weights)
            samples = np.asarray(components.means)[labels] + np.asarray(components.sigmas)[labels] * noise_rng.standard_normal(sample_count)
            prefix = f"scenario_{scenario}_level_{level}"
            populations[f"{prefix}_samples"] = samples
            populations[f"{prefix}_component_labels"] = labels
            empirical.append(dict(asdict(context), record_type="empirical_sample_statistics",
                                  population="direct prescribed-distribution draws; no channel or transition filtering",
                                  sample_count=sample_count, sample_mean=float(samples.mean()),
                                  sample_var=float(samples.var(ddof=1)), sample_std=float(samples.std(ddof=1)),
                                  variance_estimator="sample variance, ddof=1", split_d=d))
            for k in range(len(weights)):
                selected = samples[labels == k]
                empirical_components.append(dict(dataset_id=dataset_id, level_id=level, component_id=k,
                    sample_count=len(selected), sample_mean=float(selected.mean()) if len(selected) else None,
                    sample_var=float(selected.var(ddof=1)) if len(selected) > 1 else None))
            configurations.append(dict(context=asdict(context), components=asdict(components), reference=asdict(reference)))
    tables = [export_table(theoretical, output / "prescribed_level_analysis"),
              export_table(parameters, output / "component_parameters"),
              export_table(empirical, output / "empirical_statistics"),
              export_table(empirical_components, output / "empirical_component_statistics"),
              export_table([], output / "fitted_level_analysis")]
    np.savez_compressed(output / "sample_populations.npz", **populations)
    manifest = dict(output_group="internal_level_analysis", sigma=sigma, splits=list(splits), seed=seed,
                    samples_per_level_per_scenario=sample_count, sampling_phase_convention="UI after nominal symbol boundary",
                    population_kind="prescribed mixture benchmark; not a continuous waveform realization",
                    reference_kind="configured_nominal; not waveform-estimated",
                    label_rule="independent categorical draws with prescribed probabilities; labels saved in NPZ",
                    rng="SeedSequence(seed).spawn(2): labels, Gaussian component deviations",
                    fitted_status="not_available: no GMM fitter exists; no fitting performed",
                    weight_sum_tolerance=1e-12, direction_tolerance=1e-12,
                    mean_preservation_required=True, mean_preservation_tolerance=1e-12,
                    versions=dict(python=platform.python_version(), numpy=np.__version__),
                    configurations=configurations)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    configuration = configuration_table(manifest)
    configuration.write_csv(output)
    write_workbook([configuration, *tables], output)
    return theoretical, empirical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/internal_level_analysis"))
    parser.add_argument("--sigma", type=float, default=.1)
    parser.add_argument("--splits", type=float, nargs="+", default=[0., .15, .3])
    parser.add_argument("--sample-count", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    theoretical, _ = run_benchmarks(args.output, sigma=args.sigma, splits=args.splits, sample_count=args.sample_count, seed=args.seed)
    for row in theoretical:
        print(f"{row['dataset_id']} level {row['level_id']}: mu_g={row['mu_g']:.6g}, DJ_var={row['DJ_var']:.6g}, RJ_var={row['RJ_var']:.6g}, TJ_var={row['TJ_var']:.6g}, mu_abs={row['mu_abs']:.6g}")
    print(args.output)


if __name__ == "__main__":
    main()
