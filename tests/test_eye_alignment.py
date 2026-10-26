from dataclasses import replace
import numpy as np
import pytest
from dual_tail_characterization import Config, generate_waveform, extract_eye, paired_eyes
from dual_tail_characterization.plotting import eye_histogram, amplitude_limits


@pytest.mark.parametrize("levels", [(0., 1., 2., 3.), (-.05, 1.25, 2.2, 2.9)])
@pytest.mark.parametrize("before,after", [(a, b) for a in range(4) for b in range(4)])
def test_all_ordered_transitions(levels, before, after):
    c = Config(levels=levels, symbol_mode="explicit", sequence=[before]*8+[after]*8, n_symbols=16)
    eye = extract_eye(generate_waveform(c), starts=[8])
    trace = eye.traces[0]
    if before == after:
        np.testing.assert_allclose(trace, levels[before], atol=1e-12)
    else:
        fraction = (trace - levels[before]) / (levels[after] - levels[before])
        crossing = np.interp(.5, fraction, eye.relative_time_ui)
        # Generator retains its documented half-sample discrete midpoint offset.
        assert abs(crossing - .5) <= 1 / c.samples_per_ui


@pytest.mark.parametrize("phase", [0., .123, -.123])
def test_nominal_boundaries_and_source_selection(phase):
    w = generate_waveform(Config(n_symbols=12, display_phase_ui=phase))
    one, two = paired_eyes(w)
    np.testing.assert_allclose(two.start_times / w.config.symbol_period, two.clock_start_symbols - .5 + phase)
    for delta, expected in [(0, .5 - phase), (1, 1.5 - phase)]:
        positions = (two.clock_start_symbols + delta) - two.start_times / w.config.symbol_period
        np.testing.assert_allclose(positions, expected)
    np.testing.assert_array_equal(one.traces, two.traces[:, :w.config.samples_per_ui + 1])
    assert one.source_start_indices[0] >= 0
    assert two.source_start_indices[-1] + 2*w.config.samples_per_ui <= len(w.final) - 1
    with pytest.raises(ValueError, match="valid interval"):
        extract_eye(w, starts=[0])


def test_channel_crossing_shift_is_preserved():
    c = Config(n_symbols=20, symbol_mode="explicit", sequence=[0]*10+[3]*10)
    crossings = []
    for config in [c, replace(c, channel_bandwidth_hz=2e9)]:
        eye = extract_eye(generate_waveform(config), starts=[10])
        crossings.append(np.interp(1.5, eye.traces[0], eye.relative_time_ui))
    assert abs(crossings[0] - .5) < 1/c.samples_per_ui
    assert crossings[1] > crossings[0] + .25


def test_configurable_histogram_preserves_samples():
    w = generate_waveform(Config(n_symbols=20))
    eye = extract_eye(w, 2)
    counts, x, y = eye_histogram(eye, amplitude_limits(w), amplitude_bins=48, time_bins=64)
    expected, _, _ = np.histogram2d(np.tile(eye.relative_time_ui, len(eye.traces)), eye.traces.ravel(), bins=(x, y))
    np.testing.assert_array_equal(counts, expected)
    assert counts.sum() == eye.traces.size
    with pytest.raises(ValueError, match="sample phases"):
        eye_histogram(eye, amplitude_limits(w), time_bins=1000)
