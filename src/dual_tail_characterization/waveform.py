"""Centered offline logistic shaping followed by an optional causal RC channel."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve, lfilter
from scipy.special import expit
from .config import Config


@dataclass
class Waveform:
    config: Config
    time: np.ndarray
    ideal: np.ndarray
    noiseless: np.ndarray
    final: np.ndarray
    symbol_indices: np.ndarray
    symbol_amplitudes: np.ndarray
    symbol_boundaries: np.ndarray
    valid_interval: tuple[float, float]
    metadata: dict
    guard_symbol_indices: np.ndarray
    symbol_centers: np.ndarray | None = None
    symbol_endpoint_values: np.ndarray | None = None
    endpoint_standard_normals: np.ndarray | None = None


def generate_waveform(config: Config) -> Waveform:
    if config.waveform_mode == "shared_endpoints":
        from .shared_endpoints import generate_shared_endpoints
        return generate_shared_endpoints(config)
    c = config
    s = c.samples_per_ui
    dt = c.symbol_period / s
    tau_ui = c.transition_10_90_ui / (2 * np.log(9))
    # Total omitted continuous kernel mass <= tolerance before renormalization.
    half = int(np.ceil(tau_ui * np.log(2 / c.tail_tolerance - 1) * s))
    x = np.arange(-half, half + 1) / (s * tau_ui)
    kernel = expit(x) * expit(-x)
    kernel /= kernel.sum()
    channel_tau = 0. if c.channel_bandwidth_hz is None else 1 / (2 * np.pi * c.channel_bandwidth_hz)
    settling = int(np.ceil(-np.log(c.tail_tolerance) * channel_tau / dt))
    guards = int(np.ceil((half + settling) / s)) + 2
    symbols_rng, noise_rng = [np.random.default_rng(seed) for seed in np.random.SeedSequence(c.seed).spawn(2)]
    if c.symbol_mode == "random":
        all_symbols = symbols_rng.choice(len(c.levels), c.n_symbols + 2 * guards, p=c.probabilities)
    else:
        all_symbols = np.pad(np.asarray(c.sequence, dtype=int), guards, mode="edge")
    levels = np.asarray(c.levels)
    full_ideal = np.repeat(levels[all_symbols], s)
    shaped = fftconvolve(full_ideal, kernel, mode="same")
    if channel_tau:
        alpha = np.exp(-dt / channel_tau)
        b = -np.expm1(-dt / channel_tau)
        shaped, _ = lfilter([b], [1., -alpha], shaped, zi=[alpha * shaped[0]])
    start = guards * s
    stop = start + c.n_symbols * s + 1
    noiseless = shaped[start:stop].copy()
    final = noiseless + noise_rng.normal(0, c.noise_std, noiseless.size)
    indices = all_symbols[guards:guards + c.n_symbols].copy()
    time = np.arange(noiseless.size) * dt
    metadata = dict(dt_seconds=dt, kernel_half_samples=half, kernel_support_ui=half / s,
                    logistic_tau_ui=float(tau_ui), guard_symbols_per_side=guards,
                    guard_policy="independent random guards" if c.symbol_mode == "random" else "constant endpoint extension; no repetition or wraparound",
                    alignment="centered logistic kernel; ideal jumps on integer UI samples; causal channel delay retained",
                    channel_tau_seconds=channel_tau, channel_settling_samples=settling,
                    rng="SeedSequence(seed).spawn(2): symbols, noise", model="offline logistic smoothing + optional causal first-order low-pass")
    return Waveform(c, time, full_ideal[start:stop].copy(), noiseless, final, indices,
                    levels[indices], np.arange(c.n_symbols + 1) * c.symbol_period,
                    (0., c.n_symbols * c.symbol_period), metadata, all_symbols,
                    symbol_centers=(np.arange(c.n_symbols) + .5) * c.symbol_period)
