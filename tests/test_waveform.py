from dataclasses import replace
import numpy as np
import pytest
from dual_tail_characterization import Config, generate_waveform, extract_eye, paired_eyes
from dual_tail_characterization.export import save_run
from dual_tail_characterization.plotting import eye_histogram, amplitude_limits


def explicit(sequence, **kwargs):
    return Config(symbol_mode="explicit", sequence=sequence, n_symbols=len(sequence), **kwargs)


def test_reproducibility_and_stream_independence():
    c = Config(n_symbols=30, noise_std=.03)
    a, b = generate_waveform(c), generate_waveform(c)
    np.testing.assert_array_equal(a.final, b.final)
    clean = generate_waveform(replace(c, noise_std=0))
    np.testing.assert_array_equal(a.symbol_indices, clean.symbol_indices)
    np.testing.assert_array_equal(a.noiseless, clean.noiseless)
    assert not np.array_equal(a.final, clean.final)


@pytest.mark.parametrize("level", range(4))
@pytest.mark.parametrize("bandwidth", [None, 2e9])
def test_constant_noninteger(level, bandwidth):
    w = generate_waveform(explicit([level] * 8, levels=(-.05, 1.25, 2.2, 2.9), channel_bandwidth_hz=bandwidth))
    np.testing.assert_allclose(w.final, w.config.levels[level], atol=1e-12)


def test_unequal_mapping_and_timing():
    c = explicit([0, 3, 1, 2], levels=(-.05, 1.25, 2.2, 2.9))
    a, b = generate_waveform(c), generate_waveform(replace(c, symbol_period=2e-10))
    np.testing.assert_array_equal(a.symbol_amplitudes, [-.05, 2.9, 1.25, 2.2])
    np.testing.assert_allclose(np.diff(a.time), c.symbol_period / c.samples_per_ui, rtol=1e-12)
    np.testing.assert_allclose(b.time, 2 * a.time, rtol=1e-14)
    np.testing.assert_array_equal(a.final, b.final)


@pytest.mark.parametrize("rising", [True, False])
def test_isolated_transition_duration(rising):
    seq = [0] * 5 + [1] * 5 if rising else [1] * 5 + [0] * 5
    w = generate_waveform(explicit(seq, levels=(0., 1.)))
    mask = (w.time / w.config.symbol_period > 4) & (w.time / w.config.symbol_period < 6)
    y = w.final[mask] if rising else 1 - w.final[mask]
    assert np.all(np.diff(y) >= -1e-14)
    t10, t90 = np.interp([.1, .9], y, w.time[mask] / w.config.symbol_period)
    assert abs(t90 - t10 - .2) < 2 / w.config.samples_per_ui


@pytest.mark.parametrize("phase", [0., .123, -.123, 1.25])
def test_extraction_source_and_pairing(phase):
    w = generate_waveform(Config(n_symbols=14, noise_std=.02))
    one, two = paired_eyes(w, phase)
    s = w.config.samples_per_ui
    np.testing.assert_array_equal(one.traces, two.traces[:, :s + 1])
    second = extract_eye(w, 1, phase, two.clock_start_symbols + 1)
    np.testing.assert_allclose(second.traces, two.traces[:, s:], atol=1e-12)
    np.testing.assert_allclose(one.traces[:, -1], second.traces[:, 0], atol=1e-12)
    for eye in (one, two, extract_eye(w, 3, phase)):
        for start, trace in zip(eye.source_start_indices, eye.traces):
            expected = np.interp(start + np.arange(len(trace)), np.arange(len(w.final)), w.final)
            np.testing.assert_array_equal(trace, expected)
            assert start >= 0 and start + len(trace) - 1 <= len(w.final) - 1
    if phase == 0:
        for start, trace in zip(two.source_start_indices.astype(int), two.traces):
            np.testing.assert_array_equal(trace, w.final[start:start + len(trace)])
    with pytest.raises(ValueError, match="valid interval"):
        extract_eye(w, 2, starts=[13])


def test_explicit_no_wrap_and_context():
    w = generate_waveform(explicit([0, 1, 2, 3]))
    g = w.metadata["guard_symbols_per_side"]
    assert np.all(w.guard_symbol_indices[:g] == 0)
    assert np.all(w.guard_symbol_indices[-g:] == 3)
    eye = extract_eye(w, 2)
    np.testing.assert_array_equal(eye.context_symbol_indices[-1], [2, 3, -1])
    assert w.final[0] < 1e-8 and w.final[-1] > 3 - 1e-8


def test_channel_memory():
    w = generate_waveform(explicit([0] * 5 + [3] * 3, channel_bandwidth_hz=1e9))
    s = w.config.samples_per_ui
    assert w.final[6*s] < w.final[7*s] < w.final[8*s]
    assert w.final[6*s] < 2


@pytest.mark.parametrize("kwargs, message", [
    ({"levels": [1, 1]}, "levels"), ({"levels": [0, float("nan")]}, "levels"),
    ({"probabilities": [.1, .1, .1, .1]}, "probabilities"),
    ({"probabilities": [0, 0, -1, 2]}, "probabilities"),
    ({"samples_per_ui": 1}, "samples_per_ui"), ({"n_symbols": 0}, "n_symbols"),
    ({"seed": -1}, "seed"), ({"symbol_period": 0}, "symbol_period"),
    ({"transition_10_90_ui": 0}, "transition"), ({"noise_std": -1}, "noise"),
    ({"channel_bandwidth_hz": 0}, "bandwidth"), ({"display_phase_ui": float("inf")}, "phase"),
    ({"tail_tolerance": 1}, "tolerance"), ({"symbol_mode": "bad"}, "symbol_mode"),
    ({"symbol_mode": "explicit", "n_symbols": 1, "sequence": [4]}, "indices"),
    ({"symbol_mode": "explicit", "sequence": [0]}, "length"),
    ({"sequence": [0]}, "explicit"), ({"amplitude_unit": ""}, "unit")])
def test_invalid(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Config(**kwargs)


def test_export_and_raw_counts(tmp_path):
    w = generate_waveform(Config(n_symbols=20))
    save_run(w, tmp_path, csv_interval_ui=(0, 3))
    for file in ("config.json", "summary.json", "waveform.npz", "waveform.png", "eye_1ui.png", "eye_2ui.png", "waveform_interval.csv"):
        assert (tmp_path / file).stat().st_size > 0
    with np.load(tmp_path / "waveform.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["final"], w.final)
    eye = extract_eye(w)
    counts, _, _ = eye_histogram(eye, amplitude_limits(w))
    assert counts.dtype == np.int64
    assert counts.sum() == eye.traces.size
