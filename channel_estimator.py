
import argparse
import json
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.linalg import solve_toeplitz


def load_audio(path):
    """Load WAV as mono float64."""
    audio, sample_rate = sf.read(
        path,
        dtype="float64",
        always_2d=True,
    )

    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Invalid samples in: {path}")

    audio = np.mean(audio, axis=1)
    audio -= np.mean(audio)

    return audio, sample_rate


def resample_audio(audio, original_rate, target_rate):
    """Resample received audio if sample rates differ."""
    if original_rate == target_rate:
        return audio

    common = gcd(original_rate, target_rate)

    up = target_rate // common
    down = original_rate // common

    return signal.resample_poly(audio, up, down)


def align_signals(tx, rx):
    """
    Find the best temporal alignment using normalized
    cross-correlation.
    """
    if len(rx) < len(tx):
        raise ValueError(
            "Received recording is shorter than transmission."
        )

    correlation = signal.correlate(
        rx,
        tx,
        mode="valid",
        method="fft",
    )

    tx_energy = np.sum(tx ** 2)

    cumulative_energy = np.concatenate(
        ([0.0], np.cumsum(rx ** 2))
    )

    window_energy = (
        cumulative_energy[len(tx):]
        - cumulative_energy[:-len(tx)]
    )

    denominator = np.sqrt(
        tx_energy * window_energy
    )

    normalized = np.abs(correlation) / np.maximum(
        denominator,
        1e-15,
    )

    best_index = int(np.argmax(normalized))

    aligned_rx = rx[
        best_index:best_index + len(tx)
    ]

    score = float(normalized[best_index])

    return aligned_rx, best_index, score


def estimate_channel(tx, rx, taps=256, regularization=1e-3):
    """
    Estimate a causal FIR channel using regularized
    least squares and autocorrelation.
    """
    if taps < 1 or taps >= len(tx):
        raise ValueError("Invalid number of FIR taps.")

    # Autocorrelation of transmitted signal
    autocorrelation = signal.correlate(
        tx,
        tx,
        mode="full",
        method="fft",
    )

    center = len(tx) - 1

    rxx = np.real(
        autocorrelation[center:center + taps]
    )

    # Cross-correlation between received and transmitted signals
    cross_correlation = signal.correlate(
        rx,
        tx,
        mode="full",
        method="fft",
    )

    rxy = np.real(
        cross_correlation[center:center + taps]
    )

    # Regularization improves numerical stability
    rxx[0] += regularization * rxx[0]

    impulse_response = solve_toeplitz(
        (rxx, rxx),
        rxy,
    )

    # Predicted received signal
    predicted = signal.lfilter(
        impulse_response,
        [1.0],
        tx,
    )

    residual = rx - predicted

    # Ignore the initial FIR transient
    valid_start = taps

    received_power = np.mean(
        rx[valid_start:] ** 2
    )

    residual_power = np.mean(
        residual[valid_start:] ** 2
    )

    estimated_snr = 10 * np.log10(
        max(received_power, 1e-15)
        / max(residual_power, 1e-15)
    )

    # Frequency response
    frequencies, response = signal.freqz(
        impulse_response,
        worN=2048,
        fs=48000,
    )

    magnitude_db = 20 * np.log10(
        np.maximum(np.abs(response), 1e-12)
    )

    return {
        "impulse_response": impulse_response,
        "frequencies": frequencies,
        "magnitude_db": magnitude_db,
        "residual": residual,
        "estimated_snr_db": float(estimated_snr),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Acoustic channel estimator"
    )

    parser.add_argument(
        "--tx",
        default="transmission.wav",
    )

    parser.add_argument(
        "--rx",
        default="5G.wav",
    )

    parser.add_argument(
        "--report",
        default="channel_report.json",
    )

    parser.add_argument(
        "--taps",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-3,
    )

    args = parser.parse_args()

    print("\n===================================")
    print("     ACOUSTIC CHANNEL ESTIMATOR")
    print("===================================")

    tx, tx_rate = load_audio(args.tx)
    rx, rx_rate = load_audio(args.rx)

    print(f"\nTransmitted samples: {len(tx)}")
    print(f"Received samples:    {len(rx)}")
    print(f"TX sample rate:      {tx_rate} Hz")
    print(f"RX sample rate:      {rx_rate} Hz")

    if tx_rate != rx_rate:
        print("\nResampling received audio...")
        rx = resample_audio(
            rx,
            rx_rate,
            tx_rate,
        )

    # Align signals
    aligned_rx, delay, correlation = align_signals(
        tx,
        rx,
    )

    print("\n===== CHANNEL ALIGNMENT =====")
    print(f"Delay:               {delay} samples")
    print(f"Delay time:          {delay / tx_rate:.6f} seconds")
    print(f"Correlation:         {correlation:.6f}")

    # Estimate channel gain
    tx_energy = np.sum(tx ** 2)

    gain = np.dot(tx, aligned_rx) / max(
        tx_energy,
        1e-15,
    )

    print("\n===== CHANNEL GAIN =====")
    print(f"Linear gain:         {gain:.6f}")
    print(
        f"Gain in dB:          "
        f"{20 * np.log10(max(abs(gain), 1e-12)):.2f} dB"
    )

    # Estimate FIR channel
    result = estimate_channel(
        tx,
        aligned_rx,
        taps=args.taps,
        regularization=args.regularization,
    )

    impulse = result["impulse_response"]

    peak_tap = int(np.argmax(np.abs(impulse)))

    print("\n===== IMPULSE RESPONSE =====")
    print(f"FIR taps:            {len(impulse)}")
    print(f"Peak tap:            {peak_tap}")
    print(
        f"Peak tap delay:      "
        f"{peak_tap / tx_rate * 1000:.3f} ms"
    )
    print(
        f"Estimated SNR:       "
        f"{result['estimated_snr_db']:.2f} dB"
    )

    print("\n===== FREQUENCY RESPONSE =====")

    magnitude = result["magnitude_db"]

    print(f"Maximum:             {np.max(magnitude):.2f} dB")
    print(f"Minimum:             {np.min(magnitude):.2f} dB")
    print(f"Variation:           {np.ptp(magnitude):.2f} dB")

    # Save report
    report = {
        "sample_rate": tx_rate,
        "transmitted_samples": len(tx),
        "received_samples": len(rx),
        "delay_samples": delay,
        "delay_seconds": delay / tx_rate,
        "alignment_correlation": correlation,
        "channel_gain": float(gain),
        "channel_gain_db": float(
            20 * np.log10(max(abs(gain), 1e-12))
        ),
        "fir_taps": len(impulse),
        "impulse_response": impulse.tolist(),
        "frequency_hz": result["frequencies"].tolist(),
        "frequency_response_db": magnitude.tolist(),
        "estimated_snr_db": result["estimated_snr_db"],
        "regularization": args.regularization,
    }

    output = Path(args.report)

    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n===== REPORT =====")
    print(f"Saved: {output.resolve()}")
    print("\nChannel estimation completed.")


if __name__ == "__main__":
    main()
