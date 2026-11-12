from pathlib import Path
import csv

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, fftconvolve, hilbert

from config import (
    SAMPLE_RATE,
    CARRIER_FREQ,
    FFT_SIZE,
    CYCLIC_PREFIX_LENGTH,
    OFDM_SYMBOL_SAMPLES,
    ACTIVE_SUBCARRIER_INDICES,
    AUDIO_SCALE,
    TOTAL_TRAINING_SYMBOLS,
    OUTPUT_DIR,
)
from transmitter.training_signal import (
    build_training_grid,
    ofdm_modulate,
)


CAPTURE_COUNT = 3
MIN_SYNC_CONFIDENCE = 0.08
MAX_TEMPORAL_ERROR = 0.35
MAX_CROSS_CAPTURE_SHAPE_ERROR = 0.35


def _to_float_audio(audio):
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = np.mean(audio.astype(np.float64), axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        if np.issubdtype(audio.dtype, np.unsignedinteger):
            midpoint = (info.max + 1) / 2.0
            audio = (audio.astype(np.float64) - midpoint) / midpoint
        else:
            audio = audio.astype(np.float64) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float64)
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError('Audio is empty or contains NaN/Inf samples.')
    if float(np.max(np.abs(audio))) < 1e-7:
        raise ValueError('Audio recording is silent.')
    return audio


def read_audio(path):
    fs, raw = wavfile.read(path)
    if fs != SAMPLE_RATE:
        raise ValueError(f'Expected {SAMPLE_RATE} Hz, received {fs} Hz: {path}')
    return _to_float_audio(raw)


def convert_to_baseband(audio):
    """Recover complex baseband from the real passband recording."""
    audio = np.asarray(audio, dtype=np.float64)
    analytic = hilbert(audio)
    n = np.arange(audio.size, dtype=np.float64)
    mixer = np.exp(-1j * 2.0 * np.pi * CARRIER_FREQ * n / SAMPLE_RATE)
    baseband = analytic * mixer
    if not np.all(np.isfinite(baseband)):
        raise ValueError('Baseband conversion produced NaN/Inf.')
    return baseband


def build_training_reference():
    """Return the exact known complex-baseband sequence played by the TX."""
    grid = build_training_grid()
    one_symbol = ofdm_modulate(grid)
    waveform = np.tile(one_symbol, TOTAL_TRAINING_SYMBOLS)
    return grid, one_symbol, waveform * AUDIO_SCALE


def detect_single_symbol_start(baseband, one_symbol_template):
    """Coarse location of any repeated training symbol, used for CFO only."""
    corr = correlate(baseband, one_symbol_template, mode='valid', method='fft')
    energy = fftconvolve(
        np.abs(baseband) ** 2,
        np.ones(len(one_symbol_template), dtype=np.float64),
        mode='valid',
    )
    template_energy = float(np.vdot(one_symbol_template, one_symbol_template).real)
    metric = np.abs(corr) / np.sqrt(np.maximum(energy * template_energy, 1e-20))
    index = int(np.argmax(metric))
    confidence = float(metric[index])
    if not np.isfinite(confidence) or confidence < MIN_SYNC_CONFIDENCE:
        raise ValueError(f'Coarse training sync failed (metric={confidence:.4f}).')
    return index, confidence


def estimate_cfo_near(baseband, coarse_start):
    """Estimate CFO from the highest-energy adjacent repeated-symbol pairs."""
    length = OFDM_SYMBOL_SAMPLES
    radius = min(8, TOTAL_TRAINING_SYMBOLS - 1)
    candidates = []
    # Include pairs on BOTH sides of the coarse hit. A coarse detector can
    # legitimately lock to the final repetition, so following samples may be
    # silence and must not be used for the CFO estimate.
    for offset in range(-radius, radius + 1):
        first = coarse_start + offset * length
        second = first + length
        if first < 0 or second + length > baseband.size:
            continue
        a = baseband[first:first + length]
        b = baseband[second:second + length]
        score = float(np.linalg.norm(a) * np.linalg.norm(b))
        if score > 1e-12:
            candidates.append((score, a, b))

    if not candidates:
        raise ValueError('Not enough adjacent training symbols for CFO estimation.')
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:min(8, len(candidates))]
    cross_sum = sum(np.vdot(a, b) for _, a, b in selected)
    if abs(cross_sum) < 1e-15:
        raise ValueError('CFO estimation failed: repeated symbols have no energy.')
    return float(np.angle(cross_sum) * SAMPLE_RATE / (2.0 * np.pi * length))


def detect_training_start(baseband, full_template):
    """Matched-filter the ENTIRE CFO-corrected training burst."""
    baseband = np.asarray(baseband, dtype=np.complex128)
    template = np.asarray(full_template, dtype=np.complex128)
    if baseband.size < template.size:
        raise ValueError(
            f'Recording has {baseband.size} samples but the complete training '
            f'burst requires {template.size} samples.'
        )
    corr = correlate(baseband, template, mode='valid', method='fft')
    window_energy = fftconvolve(
        np.abs(baseband) ** 2,
        np.ones(template.size, dtype=np.float64),
        mode='valid',
    )
    template_energy = float(np.vdot(template, template).real)
    metric = np.abs(corr) / np.sqrt(
        np.maximum(window_energy * template_energy, 1e-20)
    )
    if not np.all(np.isfinite(metric)):
        raise ValueError('Full-burst synchronization produced invalid values.')
    start = int(np.argmax(metric))
    confidence = float(metric[start])
    if confidence < MIN_SYNC_CONFIDENCE:
        raise ValueError(
            f'Full-burst synchronization failed (confidence={confidence:.4f}). '
            'Check speaker output, microphone input and recording level.'
        )
    return start, confidence


def correct_cfo(baseband, cfo_hz):
    n = np.arange(baseband.size, dtype=np.float64)
    return baseband * np.exp(-1j * 2.0 * np.pi * cfo_hz * n / SAMPLE_RATE)


def demodulate_training_symbols(baseband, start):
    """Demodulate exactly the configured training count; never consume silence."""
    required_end = start + TOTAL_TRAINING_SYMBOLS * OFDM_SYMBOL_SAMPLES
    if required_end > baseband.size:
        available = max(0, (baseband.size - start) // OFDM_SYMBOL_SAMPLES)
        raise ValueError(
            f'Incomplete training burst: expected {TOTAL_TRAINING_SYMBOLS} '
            f'symbols, only {available} fit in the recording.'
        )

    output = np.empty((TOTAL_TRAINING_SYMBOLS, FFT_SIZE), dtype=np.complex128)
    for index in range(TOTAL_TRAINING_SYMBOLS):
        symbol_start = start + index * OFDM_SYMBOL_SAMPLES
        useful_start = symbol_start + CYCLIC_PREFIX_LENGTH
        useful = baseband[useful_start:useful_start + FFT_SIZE]
        output[index] = np.fft.fft(useful) / np.sqrt(FFT_SIZE)
    if not np.all(np.isfinite(output)):
        raise ValueError('Demodulated training symbols contain NaN/Inf.')
    return output


def _robust_complex_median(values, axis=0):
    return (
        np.median(np.real(values), axis=axis)
        + 1j * np.median(np.imag(values), axis=axis)
    )


def _estimate_one_capture(path, grid, one_symbol, full_template):
    audio = read_audio(path)
    baseband = convert_to_baseband(audio)
    coarse_start, coarse_confidence = detect_single_symbol_start(
        baseband, one_symbol * AUDIO_SCALE
    )
    cfo = estimate_cfo_near(baseband, coarse_start)
    corrected = correct_cfo(baseband, cfo)
    start, confidence = detect_training_start(corrected, full_template)
    received = demodulate_training_symbols(corrected, start)

    active_bins = np.asarray(ACTIVE_SUBCARRIER_INDICES, dtype=int) % FFT_SIZE
    transmitted = grid[active_bins] * AUDIO_SCALE
    if np.any(np.abs(transmitted) < 1e-12):
        raise ValueError('Training reference has an inactive active-subcarrier bin.')

    h_raw = received[:, active_bins] / transmitted[np.newaxis, :]

    # Remove residual common phase rotation per OFDM symbol. This does not
    # remove frequency-dependent channel phase.
    reference = _robust_complex_median(h_raw[:min(8, len(h_raw))], axis=0)
    aligned = np.empty_like(h_raw)
    symbol_phase = np.empty(len(h_raw), dtype=np.float64)
    for index, h in enumerate(h_raw):
        phase = float(np.angle(np.vdot(reference, h)))
        symbol_phase[index] = phase
        aligned[index] = h * np.exp(-1j * phase)

    channel = _robust_complex_median(aligned, axis=0)
    norm_channel = max(float(np.linalg.norm(channel)), 1e-12)
    temporal_error = np.linalg.norm(aligned - channel[None, :], axis=1) / norm_channel

    magnitude = np.abs(aligned)
    mean_magnitude = np.mean(magnitude, axis=0)
    magnitude_cv = np.std(magnitude, axis=0) / np.maximum(mean_magnitude, 1e-12)
    unit_phase = aligned / np.maximum(magnitude, 1e-12)
    phase_coherence = np.abs(np.mean(unit_phase, axis=0))

    predicted = transmitted[None, :] * channel[None, :]
    residual_power = float(np.mean(np.abs(received[:, active_bins] - predicted) ** 2))
    signal_power = float(np.mean(np.abs(predicted) ** 2))
    residual_snr = 10.0 * np.log10(
        max(signal_power, 1e-20) / max(residual_power, 1e-20)
    )

    metrics = {
        'file': path.name,
        'sync_start_sample': start,
        'sync_start_seconds': start / SAMPLE_RATE,
        'sync_confidence': confidence,
        'coarse_sync_confidence': coarse_confidence,
        'cfo_hz': cfo,
        'training_symbols_used': len(received),
        'audio_rms': float(np.sqrt(np.mean(audio ** 2))),
        'audio_peak': float(np.max(np.abs(audio))),
        'temporal_error_median': float(np.median(temporal_error)),
        'temporal_error_p95': float(np.percentile(temporal_error, 95)),
        'mean_magnitude_cv': float(np.mean(magnitude_cv)),
        'mean_phase_coherence': float(np.mean(phase_coherence)),
        'approx_residual_snr_db': float(residual_snr),
    }
    return {
        'channel': channel,
        'h_symbols': aligned,
        'temporal_error': temporal_error,
        'metrics': metrics,
    }


def _align_phase_and_timing(candidate, reference, subcarriers):
    """Align arbitrary capture phase and integer/fractional timing reference."""
    ratio = candidate * np.conj(reference)
    weights = np.abs(candidate) * np.abs(reference)
    mask = weights >= max(float(np.max(weights)) * 0.15, 1e-12)
    if np.count_nonzero(mask) < 6:
        raise ValueError('Too few reliable subcarriers for capture alignment.')

    x = np.asarray(subcarriers, dtype=np.float64)[mask]
    phase = np.unwrap(np.angle(ratio[mask]))
    w = weights[mask]
    design = np.column_stack([np.ones_like(x), x])
    root_w = np.sqrt(w / np.max(w))
    coefficients, *_ = np.linalg.lstsq(
        design * root_w[:, None], phase * root_w, rcond=None
    )
    intercept, slope = coefficients
    model = intercept + slope * np.asarray(subcarriers, dtype=np.float64)
    aligned = candidate * np.exp(-1j * model)

    # Shape error is measured after removing a scalar gain difference. The
    # unscaled aligned response is retained for the final channel average.
    alpha = np.vdot(aligned, reference) / max(np.vdot(aligned, aligned).real, 1e-20)
    shape_error = float(
        np.linalg.norm(alpha * aligned - reference)
        / max(np.linalg.norm(reference), 1e-12)
    )
    gain_ratio = float(np.linalg.norm(aligned) / max(np.linalg.norm(reference), 1e-12))
    timing_offset_samples = float(-slope * FFT_SIZE / (2.0 * np.pi))
    return aligned, shape_error, gain_ratio, timing_offset_samples


def estimate_channel_taps(channel_response):
    spectrum = np.zeros(FFT_SIZE, dtype=np.complex128)
    bins = np.asarray(ACTIVE_SUBCARRIER_INDICES, dtype=int) % FFT_SIZE
    spectrum[bins] = channel_response
    return np.fft.fftshift(np.fft.ifft(spectrum))


def _save_results(channel, taps, results, aligned_captures, differences,
                  gain_ratios, timing_offsets, stable):
    subcarriers = np.asarray(ACTIVE_SUBCARRIER_INDICES, dtype=int)
    frequencies = CARRIER_FREQ + subcarriers * SAMPLE_RATE / FFT_SIZE
    bins = subcarriers % FFT_SIZE

    np.savetxt(
        OUTPUT_DIR / 'channel_frequency_response.csv',
        np.column_stack((subcarriers, frequencies, channel.real, channel.imag,
                         np.abs(channel), np.angle(channel))),
        delimiter=',',
        header='subcarrier,frequency_hz,real,imaginary,magnitude,phase_rad',
        comments='',
    )
    np.savetxt(
        OUTPUT_DIR / 'channel_taps_interpolated.csv',
        np.column_stack((np.arange(len(taps)), taps.real, taps.imag,
                         np.abs(taps), np.abs(taps) ** 2)),
        delimiter=',',
        header='sample,real,imaginary,magnitude,power',
        comments='',
    )

    with open(OUTPUT_DIR / 'capture_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]['metrics'].keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result['metrics'])

    with open(OUTPUT_DIR / 'cross_capture_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['capture', 'shape_difference', 'gain_ratio_vs_capture_1',
                         'relative_timing_offset_samples'])
        for i, (difference, gain, timing) in enumerate(
            zip(differences, gain_ratios, timing_offsets), start=1
        ):
            writer.writerow([i, difference, gain, timing])

    np.savez(
        OUTPUT_DIR / 'channel_estimate.npz',
        channel_response=channel,
        taps=taps,
        aligned_capture_responses=aligned_captures,
        capture_differences=np.asarray(differences),
        capture_gain_ratios=np.asarray(gain_ratios),
        capture_timing_offsets_samples=np.asarray(timing_offsets),
        symbol_channel_estimates=np.asarray([r['h_symbols'] for r in results]),
        stable=bool(stable),
        active_subcarriers=subcarriers,
        acoustic_frequencies=frequencies,
    )


def run_channel_estimation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid, one_symbol, full_template = build_training_reference()
    paths = [OUTPUT_DIR / f'received_recording_{i:02d}.wav'
             for i in range(1, CAPTURE_COUNT + 1)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            'Missing recordings:\n  ' + '\n  '.join(missing)
            + '\nRun: python .\\main.py all'
        )

    print('\nFULL-BURST THREE-CAPTURE CHANNEL ESTIMATION')
    print('=' * 78)
    results = []
    for index, path in enumerate(paths, start=1):
        result = _estimate_one_capture(path, grid, one_symbol, full_template)
        results.append(result)
        m = result['metrics']
        print(f'\nCapture {index}: {path.name}')
        print(f"  Sync={m['sync_confidence']:.4f} | "
              f"start={m['sync_start_seconds']:.5f} s | "
              f"CFO={m['cfo_hz']:.3f} Hz | "
              f"symbols={m['training_symbols_used']} | "
              f"temporal error={m['temporal_error_median']:.4f} | "
              f"phase coherence={m['mean_phase_coherence']:.4f} | "
              f"residual SNR approx={m['approx_residual_snr_db']:.2f} dB")

    reference = results[0]['channel']
    subcarriers = np.asarray(ACTIVE_SUBCARRIER_INDICES, dtype=int)
    aligned = [reference.copy()]
    differences = [0.0]
    gains = [1.0]
    timing_offsets = [0.0]
    for result in results[1:]:
        h_aligned, difference, gain, timing = _align_phase_and_timing(
            result['channel'], reference, subcarriers
        )
        aligned.append(h_aligned)
        differences.append(difference)
        gains.append(gain)
        timing_offsets.append(timing)

    aligned = np.asarray(aligned)
    combined = np.mean(aligned, axis=0)
    temporal_ok = all(
        r['metrics']['temporal_error_median'] <= MAX_TEMPORAL_ERROR
        for r in results
    )
    sync_ok = all(
        r['metrics']['sync_confidence'] >= MIN_SYNC_CONFIDENCE
        and r['metrics']['training_symbols_used'] == TOTAL_TRAINING_SYMBOLS
        for r in results
    )
    capture_ok = max(differences) <= MAX_CROSS_CAPTURE_SHAPE_ERROR
    stable = bool(temporal_ok and sync_ok and capture_ok)

    print('\nCROSS-CAPTURE COMPARISON (phase + timing aligned)')
    print('-' * 78)
    for i, (difference, gain, timing) in enumerate(
        zip(differences, gains, timing_offsets), start=1
    ):
        print(f'Capture {i}: shape difference={difference:.4f} | '
              f'gain ratio={gain:.3f} | timing offset={timing:+.2f} samples')
    print(f'Temporal consistency: {"PASS" if temporal_ok else "CHECK"}')
    print(f'Full-burst sync:      {"PASS" if sync_ok else "CHECK"}')
    print(f'Capture shape:        {"PASS" if capture_ok else "CHECK"}')
    print('OVERALL RESULT:       ' +
          ('CONSISTENT CHANNEL ESTIMATE' if stable else
           'ESTIMATE SAVED — REVIEW CAPTURE METRICS'))

    taps = estimate_channel_taps(combined)
    _save_results(combined, taps, results, aligned, differences, gains,
                  timing_offsets, stable)
    print('\nSAVED OUTPUTS')
    print('-' * 78)
    for name in ('channel_estimate.npz', 'channel_frequency_response.csv',
                 'channel_taps_interpolated.csv', 'capture_metrics.csv',
                 'cross_capture_metrics.csv'):
        print(OUTPUT_DIR / name)
    print('\nNote: 512-point impulse profile is band-limited/interpolated, not 512 '
          'independent physical multipath taps.')
    return combined, taps


def run_self_tests():
    """Deterministic synthetic end-to-end tests; no audio devices required."""
    print('\nCHANNEL ESTIMATOR END-TO-END TESTS')
    print('=' * 58)
    grid, one_symbol, full_template = build_training_reference()
    rng = np.random.default_rng(20261112)
    prefix = np.zeros(2400, dtype=np.complex128)
    suffix = np.zeros(1600, dtype=np.complex128)
    tx = np.concatenate((prefix, full_template, suffix))

    # Known short multipath channel, carrier offset and low-level noise.
    channel_impulse = np.array([1.0 + 0j, 0.28 + 0.12j, 0.08 - 0.04j])
    rx = np.convolve(tx, channel_impulse)
    n = np.arange(rx.size)
    known_cfo = 2.25
    rx *= np.exp(1j * 2.0 * np.pi * known_cfo * n / SAMPLE_RATE)
    rx += (rng.standard_normal(rx.size) + 1j * rng.standard_normal(rx.size)) * 1e-4
    # Exercise the real acoustic TX/RX representation: real passband WAV ->
    # analytic signal -> complex baseband, rather than testing baseband alone.
    passband = np.real(
        rx * np.exp(1j * 2.0 * np.pi * CARRIER_FREQ * n / SAMPLE_RATE)
    )
    rx = convert_to_baseband(passband)

    coarse_start, coarse_confidence = detect_single_symbol_start(
        rx, one_symbol * AUDIO_SCALE
    )
    cfo = estimate_cfo_near(rx, coarse_start)
    corrected = correct_cfo(rx, cfo)
    start, confidence = detect_training_start(corrected, full_template)
    assert abs(start - len(prefix)) <= 8, (start, confidence, coarse_start)
    print(f'PASS: complete-burst synchronization (start={start}, metric={confidence:.4f})')

    assert abs(cfo - known_cfo) < 0.10, cfo
    print(f'PASS: CFO estimate/correction ({cfo:.3f} Hz vs {known_cfo:.3f} Hz)')

    symbols = demodulate_training_symbols(corrected, start)
    assert symbols.shape == (TOTAL_TRAINING_SYMBOLS, FFT_SIZE)
    print(f'PASS: exact training count ({symbols.shape[0]} symbols; no trailing silence)')

    # Validate LS channel recovery over active carriers. The CP protects the
    # FFT window against this short causal multipath channel.
    active = np.asarray(ACTIVE_SUBCARRIER_INDICES, dtype=int) % FFT_SIZE
    transmitted = grid[active] * AUDIO_SCALE
    h_est = symbols[:, active] / transmitted[None, :]
    expected = np.fft.fft(channel_impulse, FFT_SIZE)[active]
    # Matched-filter timing may shift the FFT window by a few samples; fit and
    # remove that deterministic linear phase before comparing channel shape.
    h_med = _robust_complex_median(h_est, axis=0)
    h_aligned, shape_error, _, _ = _align_phase_and_timing(
        h_med, expected, np.asarray(ACTIVE_SUBCARRIER_INDICES, dtype=int)
    )
    shape_error = float(np.linalg.norm(h_aligned - expected) /
                        max(np.linalg.norm(expected), 1e-12))
    assert shape_error < 0.20, shape_error
    print(f'PASS: known multipath channel recovery (relative error={shape_error:.4f})')

    # The detector must reject a recording that cannot contain the whole burst.
    try:
        detect_training_start(rx[:len(prefix) + len(one_symbol) * 4], full_template)
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete burst was incorrectly accepted.')
    print('PASS: incomplete training burst rejected')
    print('\nAll end-to-end tests passed.')
