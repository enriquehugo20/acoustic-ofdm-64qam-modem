
import sys
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import matplotlib.pyplot as plt

from config import (
    OFDM_SYMBOL_SAMPLES,
    TRAINING_SYMBOLS,
    DATA_SUBCARRIERS,
)

from receiver.audio_input import read_wav
from receiver.ofdm_sync import (
    detect_ofdm_frame_start,
    refine_ofdm_sync,
)
from receiver.ofdm_demodulator import demodulate_ofdm_symbol
from receiver.resource_grid_extractor import extract_resource_grid
from receiver.postamble_detector import detect_postamble


# ============================================================
# CONFIGURATION
# ============================================================

TRANSMITTED_FILE = PROJECT_DIR / "transmission.wav"
RECEIVED_FILE = PROJECT_DIR / "5G.wav"

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD AND SYNCHRONIZE AUDIO
# ============================================================

def load_and_synchronize(filepath: Path):

    print(f"\nLoading: {filepath.name}")

    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    audio, sample_rate = read_wav(str(filepath))
    audio = np.asarray(audio, dtype=np.float64)

    coarse_start = detect_ofdm_frame_start(audio)

    frame_start, frequency_offset = refine_ofdm_sync(
        audio,
        coarse_start,
    )

    print(f"Sample rate:       {sample_rate} Hz")
    print(f"Audio samples:     {len(audio)}")
    print(f"Coarse start:      {coarse_start}")
    print(f"Frame start:       {frame_start}")
    print(f"Frequency offset:  {frequency_offset:.4f} Hz")

    return audio, frame_start, frequency_offset


# ============================================================
# DETERMINE PAYLOAD BOUNDARY
# ============================================================

def get_payload_symbol_count(
    audio: np.ndarray,
    payload_start: int,
) -> int:

    postamble_start, score, _ = detect_postamble(
        audio,
        search_start=payload_start,
    )

    if postamble_start is not None:

        available_samples = postamble_start - payload_start

        symbol_count = int(round(
            available_samples / OFDM_SYMBOL_SAMPLES
        ))

        boundary_error = abs(
            available_samples
            - symbol_count * OFDM_SYMBOL_SAMPLES
        )

        if boundary_error <= 4:
            print(f"Postamble detected: score={score:.4f}")
            return max(0, symbol_count)

    available_samples = len(audio) - payload_start

    return max(
        0,
        available_samples // OFDM_SYMBOL_SAMPLES,
    )


# ============================================================
# EXTRACT DATA CONSTELLATION
# ============================================================

def extract_constellation(
    audio: np.ndarray,
    frame_start: int,
    frequency_offset: float,
    symbol_count: int,
) -> np.ndarray:

    header_start = (
        frame_start
        + TRAINING_SYMBOLS * OFDM_SYMBOL_SAMPLES
    )

    # Include header (one OFDM symbol) plus payload.
    total_symbols = symbol_count + 1

    constellation = []

    for index in range(total_symbols):

        symbol_start = (
            header_start
            + index * OFDM_SYMBOL_SAMPLES
        )

        received_grid = demodulate_ofdm_symbol(
            audio,
            symbol_start,
            frequency_offset,
        )

        data_symbols, _ = extract_resource_grid(received_grid)

        constellation.extend(data_symbols.tolist())

    return np.asarray(constellation, dtype=np.complex128)


# ============================================================
# GLOBAL COMPLEX GAIN ESTIMATION
# ============================================================

def compensate_global_gain(
    reference: np.ndarray,
    received: np.ndarray,
):
    """
    Estimate one complex scalar:
        received = gain * reference

    Corrects global amplitude and phase only.
    It does not equalize individual subcarriers.
    """

    denominator = np.vdot(reference, reference)

    if abs(denominator) < 1e-15:
        raise ValueError("Reference constellation has zero energy.")

    gain = np.vdot(reference, received) / denominator

    if abs(gain) < 1e-12:
        raise ValueError("Estimated global gain is near zero.")

    corrected = received / gain

    error = corrected - reference

    evm = (
        np.sqrt(
            np.mean(np.abs(error) ** 2)
            / np.mean(np.abs(reference) ** 2)
        )
        * 100.0
    )

    return gain, corrected, float(evm)


# ============================================================
# CONSTELLATION METRICS
# ============================================================

def calculate_metrics(
    reference: np.ndarray,
    received: np.ndarray,
    corrected: np.ndarray,
    gain: complex,
    evm: float,
):

    reference_power = np.mean(np.abs(reference) ** 2)
    received_power = np.mean(np.abs(received) ** 2)

    gain_db = 20 * np.log10(abs(gain))
    phase_deg = np.angle(gain, deg=True)

    # Residual before global gain compensation.
    raw_error = received - reference

    raw_evm = (
        np.sqrt(
            np.mean(np.abs(raw_error) ** 2)
            / max(reference_power, 1e-15)
        )
        * 100.0
    )

    return {
        "symbols": len(reference),
        "gain_linear": abs(gain),
        "gain_db": gain_db,
        "phase_offset_deg": phase_deg,
        "reference_power": reference_power,
        "received_power": received_power,
        "uncompensated_residual_percent": raw_evm,
        "corrected_evm_percent": evm,
    }


# ============================================================
# PLOT CONSTELLATIONS
# ============================================================

def plot_constellations(
    reference: np.ndarray,
    received: np.ndarray,
    corrected: np.ndarray,
    evm: float,
):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plots = [
        (reference, "TRANSMITTED", "Ideal reference"),
        (received, "RECEIVED", "Before global correction"),
        (
            corrected,
            "GLOBAL GAIN CORRECTED",
            f"Corrected EVM: {evm:.2f}%",
        ),
    ]

    for axis, (symbols, title, subtitle) in zip(axes, plots):

        axis.scatter(
            symbols.real,
            symbols.imag,
            s=8,
            alpha=0.45,
        )

        axis.set_title(f"{title}\n{subtitle}")
        axis.set_xlabel("In-phase (I)")
        axis.set_ylabel("Quadrature (Q)")

        axis.axhline(0, linewidth=0.6)
        axis.axvline(0, linewidth=0.6)

        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "ACOUSTIC OFDM - 64-QAM CONSTELLATION ANALYSIS",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "constellation_comparison.png"

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# EVM PER OFDM SYMBOL
# ============================================================

def plot_evm_per_ofdm_symbol(
    reference,
    received,
    gain,
):

    data_count = len(DATA_SUBCARRIERS)
    symbol_count = len(reference) // data_count

    reference_matrix = reference[:symbol_count * data_count].reshape(
        symbol_count, data_count
    )

    received_matrix = received[:symbol_count * data_count].reshape(
        symbol_count, data_count
    )

    corrected_matrix = received_matrix / gain

    evm_values = []

    for tx, rx in zip(reference_matrix, corrected_matrix):

        reference_power = np.mean(np.abs(tx) ** 2)
        error_power = np.mean(np.abs(rx - tx) ** 2)

        evm_values.append(
            np.sqrt(error_power / max(reference_power, 1e-15)) * 100.0
        )

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        np.arange(symbol_count),
        evm_values,
        marker="o",
        markersize=3,
    )

    ax.set_title("EVM PER OFDM SYMBOL")
    ax.set_xlabel("OFDM symbol index")
    ax.set_ylabel("EVM (%)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    path = OUTPUT_DIR / "evm_per_ofdm_symbol.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# RELATIVE RESPONSE PER DATA SUBCARRIER
# ============================================================

def plot_subcarrier_response(
    reference,
    received,
    gain,
):

    data_count = len(DATA_SUBCARRIERS)
    symbol_count = len(reference) // data_count

    reference_matrix = reference[:symbol_count * data_count].reshape(
        symbol_count, data_count
    )

    received_matrix = received[:symbol_count * data_count].reshape(
        symbol_count, data_count
    )

    valid = np.abs(reference_matrix) > 1e-12

    ratios = np.zeros_like(received_matrix, dtype=np.complex128)

    ratios[valid] = (
        received_matrix[valid] / reference_matrix[valid]
    )

    ratios /= gain

    response = np.mean(ratios, axis=0)

    magnitude_db = 20 * np.log10(
        np.maximum(np.abs(response), 1e-12)
    )

    phase_deg = np.angle(response, deg=True)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 8),
        sharex=True,
    )

    axes[0].plot(
        DATA_SUBCARRIERS,
        magnitude_db,
        marker="o",
    )

    axes[0].set_title("RELATIVE CHANNEL RESPONSE")
    axes[0].set_ylabel("Relative magnitude (dB)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        DATA_SUBCARRIERS,
        phase_deg,
        marker="o",
    )

    axes[1].set_xlabel("Data subcarrier index")
    axes[1].set_ylabel("Relative phase (degrees)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    path = OUTPUT_DIR / "subcarrier_response.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# I/Q ERROR DISTRIBUTION
# ============================================================

def plot_error_distribution(
    reference,
    corrected,
):

    error = corrected - reference

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(error.real, bins=60, alpha=0.8)
    axes[0].set_title("IN-PHASE ERROR")
    axes[0].set_xlabel("I error")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(error.imag, bins=60, alpha=0.8)
    axes[1].set_title("QUADRATURE ERROR")
    axes[1].set_xlabel("Q error")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    path = OUTPUT_DIR / "iq_error_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# AMPLITUDE DISTRIBUTION
# ============================================================

def plot_amplitude_comparison(
    reference,
    received,
    gain,
):

    corrected = received / gain

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.hist(
        np.abs(reference),
        bins=60,
        alpha=0.55,
        label="Transmitted",
    )

    ax.hist(
        np.abs(corrected),
        bins=60,
        alpha=0.55,
        label="Received - corrected",
    )

    ax.set_title("64-QAM SYMBOL AMPLITUDE DISTRIBUTION")
    ax.set_xlabel("Symbol magnitude")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    path = OUTPUT_DIR / "amplitude_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics: dict):

    output_path = OUTPUT_DIR / "constellation_metrics.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["metric", "value"])

        for key, value in metrics.items():
            writer.writerow([key, value])

    print(f"Metrics saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ACOUSTIC OFDM - CONSTELLATION LAB")
    print("=" * 70)

    # 1. Load and synchronize both WAV files.

    tx_audio, tx_start, tx_cfo = load_and_synchronize(
        TRANSMITTED_FILE
    )

    rx_audio, rx_start, rx_cfo = load_and_synchronize(
        RECEIVED_FILE
    )

    # 2. Determine payload symbol counts.

    tx_payload_start = (
        tx_start
        + (TRAINING_SYMBOLS + 1) * OFDM_SYMBOL_SAMPLES
    )

    rx_payload_start = (
        rx_start
        + (TRAINING_SYMBOLS + 1) * OFDM_SYMBOL_SAMPLES
    )

    tx_count = get_payload_symbol_count(
        tx_audio,
        tx_payload_start,
    )

    rx_count = get_payload_symbol_count(
        rx_audio,
        rx_payload_start,
    )

    print("\n===== SYMBOL COUNTS =====")
    print(f"Transmitted payload symbols: {tx_count}")
    print(f"Received payload symbols:    {rx_count}")

    common_count = min(tx_count, rx_count)

    if common_count <= 0:
        raise RuntimeError("No common payload OFDM symbols available.")

    print(f"Symbols selected:             {common_count}")

    # 3. Recover data constellation from both signals.

    tx_symbols = extract_constellation(
        tx_audio,
        tx_start,
        tx_cfo,
        common_count,
    )

    rx_symbols = extract_constellation(
        rx_audio,
        rx_start,
        rx_cfo,
        common_count,
    )

    if len(tx_symbols) != len(rx_symbols):
        raise RuntimeError(
            "Transmitted and received constellation lengths differ."
        )

    if len(tx_symbols) == 0:
        raise RuntimeError("No constellation symbols recovered.")

    # 4. Estimate global gain and compensate.

    gain, corrected, evm = compensate_global_gain(
        tx_symbols,
        rx_symbols,
    )

    # 5. Calculate metrics BEFORE printing them.

    metrics = calculate_metrics(
        tx_symbols,
        rx_symbols,
        corrected,
        gain,
        evm,
    )

    print("\n===== CONSTELLATION METRICS =====")
    print(f"Compared QAM symbols: {metrics['symbols']}")
    print(f"Global gain:          {metrics['gain_linear']:.6f}")
    print(f"Global gain:          {metrics['gain_db']:.3f} dB")
    print(
        f"Global phase offset:  "
        f"{metrics['phase_offset_deg']:.3f} degrees"
    )
    print(
        f"Uncompensated residual: "
        f"{metrics['uncompensated_residual_percent']:.3f}%"
    )
    print(
        f"Corrected EVM:        "
        f"{metrics['corrected_evm_percent']:.3f}%"
    )

    # 6. Generate plots and reports.

    plot_constellations(
        tx_symbols,
        rx_symbols,
        corrected,
        evm,
    )

    plot_evm_per_ofdm_symbol(
        tx_symbols,
        rx_symbols,
        gain,
    )

    plot_subcarrier_response(
        tx_symbols,
        rx_symbols,
        gain,
    )

    plot_error_distribution(
        tx_symbols,
        corrected,
    )

    plot_amplitude_comparison(
        tx_symbols,
        rx_symbols,
        gain,
    )

    save_metrics(metrics)

    np.savez_compressed(
        OUTPUT_DIR / "constellation_data.npz",
        transmitted=tx_symbols,
        received=rx_symbols,
        corrected=corrected,
        global_gain=gain,
        evm_percent=evm,
    )

    print("\n" + "=" * 70)
    print("CONSTELLATION ANALYSIS COMPLETED")
    print("=" * 70)
    print("No transmitter files modified.")
    print("No receiver files modified.")
    print("No new audio recordings generated.")


if __name__ == "__main__":
    main()
