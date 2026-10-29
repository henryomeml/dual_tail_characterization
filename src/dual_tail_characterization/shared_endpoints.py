"""Shared symbol-center amplitudes connected by endpoint-normalized logistics."""
import numpy as np
from scipy.optimize import brentq


def logistic_weight(u, kappa):
    """Algebraic tanh form of normalized expit, stable even for small kappa."""
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    u = np.asarray(u, dtype=float)
    if not np.all(np.isfinite(u)) or np.any((u < 0) | (u > 1)):
        raise ValueError("segment phase u must lie in [0, 1]")
    if kappa < 1e-6:
        return u.copy()  # linear limit; omitted correction is below 1e-14
    result = .5 * (1 + np.tanh(kappa * (u - .5) / 2) / np.tanh(kappa / 4))
    return np.where(u == 0, 0., np.where(u == 1, 1., result))


def normalized_duration(kappa):
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    if kappa < 1e-6:
        return .8
    return float(4 / kappa * np.arctanh(.8 * np.tanh(kappa / 4)))


def kappa_for_duration(duration_ui):
    if not np.isfinite(duration_ui) or not 0 < duration_ui < .8:
        raise ValueError("normalized 10-90 duration must lie in (0, 0.8) UI")
    return float(brentq(lambda k: normalized_duration(k) - duration_ui,
                        1e-8, max(1., 4 * np.log(9) / duration_ui), xtol=1e-12))


def effective_lanes(levels, sigmas):
    if len(levels) != len(sigmas):
        raise ValueError("one endpoint sigma is required per level")
    return [dict(start=a, end=b, mean_start=float(levels[a]), mean_end=float(levels[b]),
                 sigma_start=float(sigmas[a]), sigma_end=float(sigmas[b]))
            for a in range(len(levels)) for b in range(len(levels))]


def lane_statistics(mean_start, mean_end, sigma_start, sigma_end, u, kappa):
    values = np.asarray([mean_start, mean_end, sigma_start, sigma_end], dtype=float)
    if not np.all(np.isfinite(values)) or min(sigma_start, sigma_end) < 0:
        raise ValueError("lane parameters must be finite with nonnegative sigmas")
    weight = logistic_weight(u, kappa)
    mean = (1 - weight) * mean_start + weight * mean_end
    variance = (1 - weight)**2 * sigma_start**2 + weight**2 * sigma_end**2
    return mean, variance, np.sqrt(variance)


def generate_shared_endpoints(c):
    from .waveform import Waveform
    s = c.samples_per_ui
    kappa = c.shared_kappa if c.shared_kappa is not None else kappa_for_duration(c.transition_10_90_ui)
    streams = np.random.SeedSequence(c.seed).spawn(3)
    symbol_rng, endpoint_rng, _noise_rng = [np.random.default_rng(seed) for seed in streams]
    symbols = (symbol_rng.choice(len(c.levels), c.n_symbols, p=c.probabilities)
               if c.symbol_mode == "random" else np.asarray(c.sequence, dtype=int))
    nominal = np.asarray(c.levels)[symbols]
    sigmas = np.zeros(len(c.levels)) if c.endpoint_sigmas is None else np.asarray(c.endpoint_sigmas)
    z = endpoint_rng.standard_normal(c.n_symbols)
    endpoints = nominal + sigmas[symbols] * z
    return assemble_shared_waveform(c, symbols, endpoints, z)


def assemble_shared_waveform(c, symbols, endpoints, z, metadata_extra=None):
    """Common connector for already realized endpoints; never redraw amplitudes."""
    from .waveform import Waveform
    s = c.samples_per_ui
    kappa = c.shared_kappa if c.shared_kappa is not None else kappa_for_duration(c.transition_10_90_ui)
    nominal = np.asarray(c.levels)[symbols]
    sigmas = np.zeros(len(c.levels)) if c.endpoint_sigmas is None else np.asarray(c.endpoint_sigmas)
    weight = logistic_weight(np.arange(s) / s, kappa)
    # Equivalent interpolation form keeps identical endpoints exactly constant.
    segments = (endpoints[:-1, None] + weight * (endpoints[1:] - endpoints[:-1])[:, None]).ravel()
    # Each center occurs exactly once; only the very last endpoint is appended.
    final = np.concatenate((segments, endpoints[-1:]))
    time = (.5 + np.arange(len(final)) / s) * c.symbol_period
    centers = (np.arange(c.n_symbols) + .5) * c.symbol_period
    boundaries = np.arange(c.n_symbols + 1) * c.symbol_period
    ideal_indices = np.floor(.5 + np.arange(len(final)) / s).astype(int)
    ideal = nominal[ideal_indices]
    metadata = dict(model="shared symbol-center endpoints; normalized logistic segments",
                    waveform_mode="shared_endpoints", dt_seconds=c.symbol_period/s, time_origin_ui=.5,
                    effective_kappa=kappa, effective_transition_10_90_ui=normalized_duration(kappa),
                    transition_duration_convention="10-90 of endpoint-normalized L(u); solved for kappa unless explicitly supplied",
                    endpoint_sigma_unit=c.amplitude_unit, endpoint_sigmas=sigmas.tolist(),
                    alignment="endpoints at symbol centers (k+0.5) UI; boundaries at integer UI; no crossing recentering",
                    valid_interval_kind="first through last symbol center, inclusive; no artificial endpoint guards",
                    guard_symbols_per_side=0, guard_policy="none; no repetition or wraparound",
                    rng="SeedSequence(seed).spawn(3): symbols, endpoint fluctuations, reserved additive noise",
                    analytic_reference="pre-impairment independent endpoint fluctuations; channel and additive noise disabled",
                    segment_dependence="adjacent segments share endpoints and are correlated",
                    continuity="amplitude continuity only; derivatives need not agree at joins")
    metadata.update(metadata_extra or {})
    return Waveform(c, time, ideal, final.copy(), final, symbols.copy(), nominal,
                    boundaries, (float(time[0]), float(time[-1])), metadata, np.array([], dtype=int),
                    centers, endpoints, z)
