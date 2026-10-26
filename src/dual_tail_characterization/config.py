from dataclasses import dataclass, asdict
import math
import numbers
import numpy as np


@dataclass(frozen=True)
class Config:
    levels: tuple[float, ...] = (0., 1., 2., 3.)
    symbol_period: float = 100e-12
    samples_per_ui: int = 128
    n_symbols: int = 10_000
    seed: int = 123
    symbol_mode: str = "random"
    sequence: tuple[int, ...] | None = None
    probabilities: tuple[float, ...] | None = None
    transition_10_90_ui: float = .20
    channel_bandwidth_hz: float | None = None
    noise_std: float = 0.
    display_phase_ui: float = 0.  # start = boundary - .5 UI + phase; positive shifts boundaries left
    amplitude_unit: str = "arbitrary units"
    tail_tolerance: float = 1e-10

    def __post_init__(self):
        levels = np.asarray(self.levels, dtype=float)
        if levels.ndim != 1 or len(levels) < 2 or not np.all(np.isfinite(levels)) or not np.all(np.diff(levels) > 0):
            raise ValueError("levels must be finite, strictly increasing, and contain at least two values")
        object.__setattr__(self, "levels", tuple(levels.tolist()))
        for name, minimum in (("samples_per_ui", 2), ("n_symbols", 1), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("symbol_period", "transition_10_90_ui"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.tail_tolerance < .01:
            raise ValueError("tail_tolerance must be between 0 and .01")
        if self.channel_bandwidth_hz is not None and (not math.isfinite(self.channel_bandwidth_hz) or self.channel_bandwidth_hz <= 0):
            raise ValueError("channel_bandwidth_hz must be finite and positive")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        if not math.isfinite(self.display_phase_ui):
            raise ValueError("display_phase_ui must be finite")
        if not isinstance(self.amplitude_unit, str) or not self.amplitude_unit.strip():
            raise ValueError("amplitude_unit must be a nonempty string")
        if self.symbol_mode not in ("random", "explicit"):
            raise ValueError("symbol_mode must be random or explicit")
        if self.probabilities is not None:
            p = np.asarray(self.probabilities, dtype=float)
            if p.shape != levels.shape or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1., rtol=0, atol=1e-12):
                raise ValueError("probabilities must match levels, be nonnegative, and sum to one")
            object.__setattr__(self, "probabilities", tuple(p.tolist()))
        if self.symbol_mode == "explicit":
            if self.sequence is None or len(self.sequence) != self.n_symbols:
                raise ValueError("explicit sequence length must equal n_symbols")
            if any(isinstance(s, bool) or not isinstance(s, numbers.Integral) or not 0 <= s < len(levels) for s in self.sequence):
                raise ValueError("sequence indices must be integers within the levels array")
            object.__setattr__(self, "sequence", tuple(int(s) for s in self.sequence))
            if self.probabilities is not None:
                raise ValueError("probabilities apply only to random symbols")
        elif self.sequence is not None:
            raise ValueError("sequence requires explicit symbol_mode")

    def to_dict(self):
        return asdict(self)
