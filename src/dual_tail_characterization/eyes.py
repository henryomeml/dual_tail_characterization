"""Linear interpolation at clock starts + display phase; endpoint inclusive."""
from dataclasses import dataclass
import numbers
import numpy as np
from .waveform import Waveform


@dataclass
class Eye:
    relative_time_ui: np.ndarray
    traces: np.ndarray
    start_times: np.ndarray
    source_start_indices: np.ndarray
    clock_start_symbols: np.ndarray
    context_symbol_indices: np.ndarray
    context_symbol_positions: np.ndarray


def extract_eye(waveform: Waveform, window_ui: int = 1, phase_ui=None, starts=None) -> Eye:
    if isinstance(window_ui, bool) or not isinstance(window_ui, numbers.Integral) or window_ui <= 0:
        raise ValueError("window_ui must be a positive integer")
    c = waveform.config
    phase = c.display_phase_ui if phase_ui is None else phase_ui
    if not np.isfinite(phase):
        raise ValueError("phase_ui must be finite")
    if starts is None:
        starts = np.arange(int(np.ceil(-phase)), int(np.floor(c.n_symbols - window_ui - phase)) + 1)
    starts = np.asarray(starts)
    if starts.ndim != 1 or not np.all(np.isfinite(starts)) or not np.all(starts == np.floor(starts)):
        raise ValueError("starts must contain integer clock positions")
    starts = starts.astype(int)
    first = (starts + phase) * c.samples_per_ui
    last = first + window_ui * c.samples_per_ui
    if np.any(first < 0) or np.any(last > len(waveform.final) - 1):
        raise ValueError("requested eye window crosses valid interval")
    relative = np.arange(window_ui * c.samples_per_ui + 1) / c.samples_per_ui
    positions = first[:, None] + np.arange(len(relative))
    traces = np.interp(positions, np.arange(len(waveform.final)), waveform.final)
    context_positions = np.floor(starts[:, None] + phase).astype(int) + np.arange(window_ui + 1)
    labels = np.full(context_positions.shape, -1, dtype=int)
    valid = (context_positions >= 0) & (context_positions < c.n_symbols)
    labels[valid] = waveform.symbol_indices[context_positions[valid]]
    return Eye(relative, traces, first * waveform.metadata["dt_seconds"], first,
               starts, labels, context_positions)


def paired_eyes(waveform: Waveform, phase_ui=None):
    two = extract_eye(waveform, 2, phase_ui)
    one = extract_eye(waveform, 1, phase_ui, two.clock_start_symbols)
    return one, two
