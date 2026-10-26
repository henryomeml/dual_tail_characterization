"""Whole-level amplitude-mixture diagnostics, independent of eye boundaries.

DJ/RJ here name between-component and within-component amplitude variance.
They do not establish physical deterministic/random sources or timing jitter.
No tail truncation, mixture fitting, quantiles, EH or BER is computed here.
"""
from dataclasses import dataclass
import math
import numbers
import numpy as np


def _finite(value):
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _window_valid(window):
    return isinstance(window, (tuple, list, np.ndarray)) and np.ndim(window) == 1 and len(window) == 2 and all(_finite(v) for v in window) and window[0] <= window[1]


@dataclass(frozen=True)
class Components:
    weights: tuple[float, ...]
    means: tuple[float, ...]
    sigmas: tuple[float, ...]
    weight_sum_tolerance: float = 1e-12
    nominal_amplitude: float | None = None
    offsets: tuple[float, ...] | None = None

    def __post_init__(self):
        if not _finite(self.weight_sum_tolerance) or not 0 <= self.weight_sum_tolerance < 1:
            raise ValueError("weight_sum_tolerance must be finite and in [0, 1)")
        for name in ("weights", "means", "sigmas"):
            values = np.asarray(getattr(self, name))
            if values.ndim != 1 or not len(values) or any(not _finite(v) for v in values):
                raise ValueError(f"{name} must be a nonempty finite numeric vector")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if not len(self.weights) == len(self.means) == len(self.sigmas):
            raise ValueError("weights, means and sigmas must have equal lengths")
        if min(self.weights) < 0:
            raise ValueError("weights must be nonnegative")
        try:
            weight_sum = math.fsum(self.weights)
        except OverflowError as exc:
            raise ValueError("weights must sum to one within weight_sum_tolerance") from exc
        if abs(weight_sum - 1) > self.weight_sum_tolerance:
            raise ValueError("weights must sum to one within weight_sum_tolerance; no normalization is performed")
        if min(self.sigmas) < 0:
            raise ValueError("sigmas must be nonnegative")
        if self.nominal_amplitude is not None and not _finite(self.nominal_amplitude):
            raise ValueError("nominal_amplitude must be finite")
        if self.offsets is not None:
            if self.nominal_amplitude is None or len(self.offsets) != len(self.weights) or any(not _finite(x) for x in self.offsets):
                raise ValueError("offsets require a finite nominal amplitude and one finite offset per component")
            if tuple(self.nominal_amplitude + x for x in self.offsets) != self.means:
                raise ValueError("means must equal nominal_amplitude + offsets")
            object.__setattr__(self, "offsets", tuple(float(x) for x in self.offsets))

    @classmethod
    def from_offsets(cls, nominal_amplitude, weights, offsets, sigmas, *,
                     require_mean_preservation=False, mean_tolerance=1e-12,
                     weight_sum_tolerance=1e-12):
        if not _finite(nominal_amplitude):
            raise ValueError("nominal_amplitude must be finite")
        if not _finite(mean_tolerance) or mean_tolerance < 0:
            raise ValueError("mean_tolerance must be finite and nonnegative")
        offsets = tuple(offsets)
        if not offsets or any(not _finite(x) for x in offsets):
            raise ValueError("offsets must be nonempty and finite")
        result = cls(weights, tuple(nominal_amplitude + x for x in offsets), sigmas,
                     weight_sum_tolerance, nominal_amplitude, tuple(offsets))
        if require_mean_preservation and abs(math.fsum(w*d for w, d in zip(result.weights, result.offsets))) > mean_tolerance:
            raise ValueError("mean-preserving offsets require sum(weights * offsets) = 0 within mean_tolerance")
        return result


@dataclass(frozen=True)
class LevelContext:
    dataset_id: str
    level_id: str | int
    amplitude_unit: str
    sampling_phase_ui: float
    sampling_window_ui: tuple[float, float] | None = None
    parameter_origin: str = "prescribed_simulation"
    channel_id: str | None = None
    wavelength_nm: float | None = None

    def __post_init__(self):
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be nonempty")
        if isinstance(self.level_id, bool) or not isinstance(self.level_id, (str, int)) or self.level_id == "":
            raise ValueError("level_id must be a nonempty string or integer")
        if not isinstance(self.amplitude_unit, str) or not self.amplitude_unit.strip():
            raise ValueError("amplitude_unit must be nonempty")
        if not _finite(self.sampling_phase_ui):
            raise ValueError("sampling_phase_ui must be finite")
        if self.sampling_window_ui is not None:
            window = self.sampling_window_ui
            if not _window_valid(window):
                raise ValueError("sampling_window_ui must be finite and ordered")
            object.__setattr__(self, "sampling_window_ui", tuple(float(x) for x in window))
        if self.parameter_origin not in ("prescribed_simulation", "fitted_measurement"):
            raise ValueError("parameter_origin must be prescribed_simulation or fitted_measurement")
        if self.channel_id is not None and (not isinstance(self.channel_id, str) or not self.channel_id.strip()):
            raise ValueError("channel_id must be a nonempty string when supplied")
        if self.wavelength_nm is not None and (not _finite(self.wavelength_nm) or self.wavelength_nm <= 0):
            raise ValueError("wavelength_nm must be finite and positive")


@dataclass(frozen=True)
class Reference:
    mu_ref: float | None = None
    sigma_ref: float | None = None
    reference_kind: str = "user_supplied"
    validity: bool = True
    reason: str | None = None
    selection_rule: str | None = None
    sampling_phase_ui: float | None = None
    sampling_window_ui: tuple[float, float] | None = None
    sample_count: int | None = None
    estimation_method: str | None = None

    def status(self):
        # Reference failures invalidate only reference outputs, not decomposition.
        if self.validity is not True:
            return "invalid", self.reason or "Reference explicitly marked invalid"
        if not _finite(self.mu_ref):
            return "invalid", "Reference mean is missing or nonfinite"
        if self.sigma_ref is not None and (not _finite(self.sigma_ref) or self.sigma_ref < 0):
            return "invalid", "Reference standard deviation must be finite and nonnegative"
        if self.reference_kind not in ("configured_nominal", "steady_state_lane", "user_supplied"):
            return "invalid", "Unknown reference_kind"
        if self.reference_kind == "steady_state_lane":
            if not self.selection_rule or not self.estimation_method or not _finite(self.sampling_phase_ui):
                return "invalid", "Lane reference requires selection rule, phase and estimation method"
            window = self.sampling_window_ui
            if not _window_valid(window):
                return "invalid", "Lane reference requires a finite ordered sampling window"
            if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, numbers.Integral) or self.sample_count < 1:
                return "invalid", "Lane reference requires a positive sample count"
        return "valid", None


def estimate_lane_reference(samples, *, selection_rule, sampling_phase_ui,
                            sampling_window_ui, min_samples=2):
    """Estimate from an already selected lane, never from the largest component.

    The caller owns the selection rule. All supplied samples must be finite;
    invalid/insufficient populations return an invalid Reference, not zero.
    """
    if isinstance(min_samples, bool) or not isinstance(min_samples, numbers.Integral) or min_samples < 2:
        raise ValueError("min_samples must be an integer >= 2")
    x = np.asarray(samples, dtype=float)
    provenance = dict(reference_kind="steady_state_lane", selection_rule=selection_rule,
                      sampling_phase_ui=sampling_phase_ui, sampling_window_ui=sampling_window_ui,
                      sample_count=x.size, estimation_method="arithmetic_mean; sample_std_ddof_1")
    if x.ndim != 1 or x.size < min_samples or not np.all(np.isfinite(x)):
        return Reference(validity=False, reason="Lane must be a finite vector with sufficient selected samples", **provenance)
    return Reference(mu_ref=float(x.mean()), sigma_ref=float(x.std(ddof=1)), **provenance)


def analyze_level(components: Components, context: LevelContext, reference: Reference | None = None,
                  *, direction_tolerance=1e-12):
    """Return an internal diagnostic row. Direction tolerance flags, never clips.

    Accepted weights are used as supplied (including tolerance-sized sum errors).
    Zero sigmas are allowed; a future fitter can apply its own positive floor.
    """
    if not _finite(direction_tolerance) or direction_tolerance < 0:
        raise ValueError("direction_tolerance must be finite and nonnegative")
    from dataclasses import asdict
    try:
        mu_g = math.fsum(w*m for w, m in zip(components.weights, components.means) if w > 0)
        DJ_var = math.fsum(w*(m-mu_g)**2 for w, m in zip(components.weights, components.means) if w > 0)
        RJ_var = math.fsum(w*s**2 for w, s in zip(components.weights, components.sigmas) if w > 0)
        TJ_var = DJ_var + RJ_var
    except (OverflowError, ValueError) as exc:
        raise ValueError("Derived level metrics exceed finite numeric range") from exc
    if not all(math.isfinite(v) for v in (mu_g, DJ_var, RJ_var, TJ_var)):
        raise ValueError("Derived level metrics exceed finite numeric range")
    row = dict(asdict(context), output_group="internal_level_analysis", component_count=len(components.weights), mu_g=mu_g,
               DJ_var=DJ_var, RJ_var=RJ_var, TJ_var=TJ_var,
               sigma_DJ=math.sqrt(DJ_var), sigma_RJ=math.sqrt(RJ_var), sigma_TJ=math.sqrt(TJ_var),
               variance_unit=f"({context.amplitude_unit})**2", rms_deviation_unit=context.amplitude_unit,
               weight_sum=math.fsum(components.weights), weight_sum_tolerance=components.weight_sum_tolerance,
               weights_policy="used as supplied; no normalization", direction_tolerance=direction_tolerance,
               mu_ref=None, sigma_ref=None, delta_g=None, mu_abs=None,
               mu_abs_name="Reference-based absolute DJ mean",
               reference_kind=reference.reference_kind if reference is not None else None,
               reference_validity=False, reference_status="missing", reference_reason="No reference supplied",
               reference_provenance=None, direction_status="unavailable")
    if reference is not None:
        status, reason = reference.status()
        row.update(reference_status=status, reference_validity=status == "valid", reference_reason=reason,
                   reference_provenance={k: v for k, v in asdict(reference).items() if k not in ("mu_ref", "sigma_ref")})
        if status == "valid":
            delta = mu_g - reference.mu_ref
            sign = 0 if delta == 0 else 1 if delta > 0 else -1
            direction = "exact_zero" if sign == 0 else ("near_zero_positive" if sign > 0 else "near_zero_negative") if abs(delta) <= direction_tolerance else ("positive" if sign > 0 else "negative")
            marker = reference.mu_ref + sign * row["sigma_DJ"]
            if not math.isfinite(delta) or not math.isfinite(marker):
                row.update(reference_status="invalid", reference_validity=False, reference_reason="Reference arithmetic exceeds finite numeric range")
            else:
                row.update(mu_ref=float(reference.mu_ref), sigma_ref=None if reference.sigma_ref is None else float(reference.sigma_ref),
                           delta_g=delta, mu_abs=marker, direction_status=direction)
    return row


def component_records(components: Components, context: LevelContext):
    """Long-form component parameters, exported separately from diagnostic rows."""
    from dataclasses import asdict
    return [dict(asdict(context), component_id=k, weight=w, mu=m, sigma=s,
                 nominal_amplitude=components.nominal_amplitude,
                 offset=None if components.offsets is None else components.offsets[k])
            for k, (w, m, s) in enumerate(zip(components.weights, components.means, components.sigmas))]
