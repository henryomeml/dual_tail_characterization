from dataclasses import replace
import json
import numpy as np
import pytest
from dual_tail_characterization.level_analysis import (
    Components, LevelContext, Reference, analyze_level, component_records, estimate_lane_reference,
)
from dual_tail_characterization.level_analysis_export import export_table
from dual_tail_characterization.level_benchmarks import run_benchmarks


def context(**kwargs):
    return LevelContext("analytic", 0, "V", .5, (.45, .55), **kwargs)


def test_single_component_and_units():
    row = analyze_level(Components([1], [1.25], [.2]), context())
    assert row["DJ_var"] == 0
    assert row["RJ_var"] == pytest.approx(.04)
    assert row["sigma_RJ"] == .2
    assert row["variance_unit"] == "(V)**2"
    assert row["rms_deviation_unit"] == "V"


@pytest.mark.parametrize("d", [0., .15, .3, 2.])
def test_symmetric_split_exact_marker(d):
    c = Components.from_offsets(0., [.5, .5], [-d, d], [.1, .1], require_mean_preservation=True)
    row = analyze_level(c, context(), Reference(0., reference_kind="configured_nominal"))
    assert row["mu_g"] == 0
    assert row["DJ_var"] == d**2
    assert row["RJ_var"] == .1**2
    assert row["TJ_var"] == .1**2 + d**2
    assert row["sigma_DJ"] == d
    assert row["sigma_TJ"] == np.sqrt(.1**2 + d**2)
    assert row["delta_g"] == row["mu_abs"] == 0
    assert row["direction_status"] == "exact_zero"
    assert row["reference_kind"] == "configured_nominal"


def test_broadening_and_weighted_variance():
    a = analyze_level(Components([1], [3], [.1]), context())
    b = analyze_level(Components([1], [3], [.2]), context())
    assert b["RJ_var"] == 4*a["RJ_var"]
    row = analyze_level(Components([.25, .75], [0, 1], [1, 2]), context())
    assert row["RJ_var"] == .25 + .75*4  # not the unweighted sum 5


def test_translation_scale_permutation_and_total_identity():
    c = Components([.25, .5, .25], [-1., 1., 3.], [.1, .2, .3])
    ref = Reference(.25)
    row = analyze_level(c, context(), ref)
    translated = analyze_level(replace(c, means=tuple(m+7 for m in c.means)), context(), Reference(7.25))
    scaled = analyze_level(replace(c, means=tuple(m*3 for m in c.means), sigmas=tuple(s*3 for s in c.sigmas)), context(), Reference(.75))
    permuted = analyze_level(Components(c.weights[::-1], c.means[::-1], c.sigmas[::-1]), context(), ref)
    for key in ("DJ_var", "RJ_var", "TJ_var"):
        assert translated[key] == pytest.approx(row[key])
        assert scaled[key] == pytest.approx(9*row[key])
    assert translated["delta_g"] == row["delta_g"]
    for key in ("mu_g", "mu_abs"):
        assert translated[key] == pytest.approx(row[key]+7)
    for key in ("sigma_DJ", "sigma_RJ", "sigma_TJ", "delta_g", "mu_abs", "mu_g"):
        assert scaled[key] == pytest.approx(3*row[key])
    assert permuted == row
    expected_second_moment = sum(w*(m*m+s*s) for w, m, s in zip(c.weights, c.means, c.sigmas))
    assert row["TJ_var"] == pytest.approx(expected_second_moment - row["mu_g"]**2)
    assert row["TJ_var"] == row["DJ_var"] + row["RJ_var"]


@pytest.mark.parametrize("reference", [None, Reference(), Reference(float("nan")),
    Reference(0., sigma_ref=-1), Reference(0., sigma_ref=float("inf")),
    Reference(0., validity=False, reason="lane not steady"),
    Reference(0., reference_kind="steady_state_lane"), Reference(0., reference_kind="unknown")])
def test_missing_invalid_references_do_not_discard_decomposition(reference):
    row = analyze_level(Components([.5, .5], [-1, 1], [.1, .1]), context(), reference)
    assert row["DJ_var"] == 1
    for key in ("mu_ref", "sigma_ref", "delta_g", "mu_abs"):
        assert row[key] is None
    assert not row["reference_validity"]
    assert row["reference_reason"]
    assert row["direction_status"] == "unavailable"


@pytest.mark.parametrize("delta,status", [(0., "exact_zero"), (1e-9, "near_zero_positive"),
    (-1e-9, "near_zero_negative"), (1e-3, "positive"), (-1e-3, "negative")])
def test_direction_tolerance_flags_without_changing_formula(delta, status):
    row = analyze_level(Components([.5, .5], [-1+delta, 1+delta], [0, 0]), context(), Reference(0.), direction_tolerance=1e-8)
    assert row["direction_status"] == status
    expected = np.sign(row["delta_g"]) * row["sigma_DJ"]
    assert row["mu_abs"] == expected
    if delta:
        assert abs(row["mu_abs"]) > .99  # tolerance must not replace sign by zero


def test_zero_sigmas_zero_weights_and_reference_sigma_metadata():
    c = Components([1., 0.], [1.25, 1e300], [0., 1e300])
    row = analyze_level(c, context(), Reference(1.25, sigma_ref=99.))
    assert row["TJ_var"] == row["sigma_TJ"] == 0
    assert row["sigma_ref"] == 99
    assert "absolute_RJ" not in row


def test_offsets_mean_preservation_option():
    with pytest.raises(ValueError, match="mean-preserving"):
        Components.from_offsets(1., [.2, .8], [-1, 1], [0, 0], require_mean_preservation=True)
    c = Components.from_offsets(1., [.2, .8], [-1, 1], [0, 0])
    assert analyze_level(c, context())["mu_g"] == pytest.approx(1.6)
    rows = component_records(c, context())
    assert rows[0]["offset"] == -1
    assert rows[0]["nominal_amplitude"] == 1


@pytest.mark.parametrize("kwargs,match", [
    ({"weights": [-.1, 1.1]}, "nonnegative"), ({"weights": [.2, .2]}, "sum to one"),
    ({"weights": [float("nan"), 1]}, "finite"), ({"weights": []}, "nonempty"),
    ({"weights": [[.5, .5]]}, "vector"), ({"weights": [True, False]}, "numeric"),
    ({"means": [0]}, "equal lengths"), ({"means": [0, float("inf")]}, "finite"),
    ({"sigmas": [-1, 1]}, "nonnegative"), ({"sigmas": [float("nan"), 1]}, "finite"),
    ({"weight_sum_tolerance": -1}, "tolerance"), ({"weight_sum_tolerance": float("nan")}, "tolerance")])
def test_invalid_components(kwargs, match):
    values = dict(weights=[.5, .5], means=[0., 1.], sigmas=[.1, .1])
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        Components(**values)


def test_tolerated_weights_are_not_normalized():
    c = Components([.5, .5+1e-13], [0., 1.], [0., 0.])
    assert c.weights[1] == .5+1e-13
    assert analyze_level(c, context())["mu_g"] == .5+1e-13


@pytest.mark.parametrize("kwargs,match", [
    ({"dataset_id": ""}, "dataset_id"), ({"amplitude_unit": ""}, "amplitude_unit"),
    ({"sampling_phase_ui": float("nan")}, "sampling_phase"),
    ({"sampling_window_ui": (1, 0)}, "window"), ({"parameter_origin": "unknown"}, "parameter_origin"),
    ({"wavelength_nm": -1}, "wavelength"), ({"channel_id": ""}, "channel_id")])
def test_invalid_context(kwargs, match):
    values = dict(dataset_id="test", level_id=0, amplitude_unit="V", sampling_phase_ui=.5)
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        LevelContext(**values)


def test_lane_reference_provenance_and_failure():
    kwargs = dict(selection_rule="known symbol 1, run length >= 8; samples already selected", sampling_phase_ui=.5, sampling_window_ui=(.45, .55))
    ref = estimate_lane_reference([.9, 1., 1.1], **kwargs)
    row = analyze_level(Components([1], [1.2], [.1]), context(), ref)
    assert row["reference_validity"]
    assert row["mu_ref"] == 1
    assert row["sigma_ref"] == pytest.approx(.1)
    assert row["reference_provenance"]["sample_count"] == 3
    for samples in ([], [1], [1, float("nan")]):
        bad = estimate_lane_reference(samples, **kwargs)
        assert analyze_level(Components([1], [1], [0]), context(), bad)["mu_abs"] is None
    incomplete = replace(ref, selection_rule=None)
    assert not analyze_level(Components([1], [1], [0]), context(), incomplete)["reference_validity"]


def test_strict_export_nulls_and_optional_metadata(tmp_path):
    row = analyze_level(Components([1], [1], [.1]), context(channel_id="ch1", wavelength_nm=1550.), Reference(float("nan")))
    export_table([row], tmp_path / "internal")
    decoded = json.loads((tmp_path / "internal.json").read_text())
    assert decoded[0]["mu_abs"] is None
    assert decoded[0]["wavelength_nm"] == 1550
    assert "NaN" not in (tmp_path / "internal.json").read_text()
    assert "mu_abs" in (tmp_path / "internal.csv").read_text()


def test_prescribed_and_empirical_benchmarks(tmp_path):
    theoretical, empirical = run_benchmarks(tmp_path, splits=(.3,), sample_count=40_000)
    assert theoretical[0]["DJ_var"] == .3**2
    assert theoretical[3]["RJ_var"] == .2**2
    for theory, sample in zip(theoretical, empirical):
        # Statistical tolerances for this seeded population; no fitted recovery claim.
        assert sample["sample_mean"] == pytest.approx(theory["mu_g"], abs=.007)
        assert sample["sample_var"] == pytest.approx(theory["TJ_var"], rel=.04)
    with np.load(tmp_path / "sample_populations.npz", allow_pickle=False) as data:
        assert data["scenario_0_level_0_component_labels"].shape == (40_000,)
    assert json.loads((tmp_path / "fitted_level_analysis.json").read_text()) == []
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["fitted_status"].startswith("not_available")


def test_invalid_analysis_controls():
    with pytest.raises(ValueError, match="direction_tolerance"):
        analyze_level(Components([1], [0], [0]), context(), direction_tolerance=-1)
    with pytest.raises(ValueError, match="mean_tolerance"):
        Components.from_offsets(0., [1], [0], [0], mean_tolerance=-1)
    with pytest.raises(ValueError, match="offsets"):
        Components.from_offsets(0., [1], [float("nan")], [0])
