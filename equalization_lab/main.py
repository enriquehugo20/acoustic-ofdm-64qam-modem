
import csv
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    PROJECT_DIR.parent
    / "channel_estimation_lab"
    / "output"
)

CHANNEL_FILE = OUTPUT_DIR / "channel_estimate.npz"
CONSISTENCY_FILE = OUTPUT_DIR / "channel_consistency.npz"

EPSILON = 1e-12

# Normalized average symbol energy for our 64-QAM constellation.
SYMBOL_ENERGY = 1.0

# Illustrative noise variance levels, relative to Es = 1.
NOISE_LEVELS_DB = [-30, -25, -20, -15, -10, -5, 0]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not CHANNEL_FILE.exists():
        raise FileNotFoundError(CHANNEL_FILE)

    if not CONSISTENCY_FILE.exists():
        raise FileNotFoundError(CONSISTENCY_FILE)

    with np.load(CHANNEL_FILE) as source:
        channel_data = {
            key: source[key].copy()
            for key in source.files
        }

    with np.load(CONSISTENCY_FILE) as source:
        consistency_data = {
            key: source[key].copy()
            for key in source.files
        }

    required_channel = [
        "channel_response",
        "taps",
        "aligned_capture_responses",
        "capture_differences",
        "capture_gain_ratios",
        "capture_timing_offsets_samples",
        "symbol_channel_estimates",
        "stable",
        "active_subcarriers",
        "acoustic_frequencies",
    ]

    for key in required_channel:
        if key not in channel_data:
            raise KeyError(f"Missing channel data: {key}")

    required_consistency = [
        "channel_estimates",
        "magnitude_std",
        "magnitude_cv",
        "phase_std_deg",
        "temporal_coherence",
    ]

    for key in required_consistency:
        if key not in consistency_data:
            raise KeyError(f"Missing consistency data: {key}")

    return channel_data, consistency_data


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_data(channel_data, consistency_data):

    channel = channel_data["channel_response"]
    taps = channel_data["taps"]
    subcarriers = channel_data["active_subcarriers"]
    frequencies = channel_data["acoustic_frequencies"]

    if channel.ndim != 1:
        raise ValueError("Invalid channel_response dimensions.")

    if taps.ndim != 1:
        raise ValueError("Invalid taps dimensions.")

    if not (
        channel.shape == subcarriers.shape == frequencies.shape
    ):
        raise ValueError("Channel arrays have inconsistent dimensions.")

    if not np.all(np.isfinite(channel)):
        raise ValueError("Channel contains invalid values.")

    if not np.all(np.isfinite(taps)):
        raise ValueError("Taps contain invalid values.")

    estimates = channel_data["symbol_channel_estimates"]

    if estimates.ndim != 3:
        raise ValueError("Invalid symbol_channel_estimates dimensions.")

    if estimates.shape[2] != len(channel):
        raise ValueError("Symbol estimates do not match channel size.")

    if consistency_data["phase_std_deg"].shape != channel.shape:
        raise ValueError("Phase consistency array has invalid dimensions.")

    if consistency_data["magnitude_cv"].shape != channel.shape:
        raise ValueError("Magnitude CV array has invalid dimensions.")


# ============================================================
# CHANNEL RESPONSE ANALYSIS
# ============================================================

def analyze_channel(channel_data, consistency_data):

    channel = channel_data["channel_response"]
    subcarriers = channel_data["active_subcarriers"]
    frequencies = channel_data["acoustic_frequencies"]

    magnitude = np.abs(channel)
    magnitude_db = 20 * np.log10(
        np.maximum(magnitude, EPSILON)
    )

    phase_deg = np.rad2deg(np.unwrap(np.angle(channel)))

    strongest = int(np.argmax(magnitude))
    weakest = int(np.argmin(magnitude))

    print("\n" + "=" * 76)
    print("CHANNEL FREQUENCY RESPONSE")
    print("=" * 76)

    print(f"Active subcarriers:       {len(channel)}")
    print(f"Mean magnitude:           {np.mean(magnitude_db):.2f} dB")
    print(f"Median magnitude:         {np.median(magnitude_db):.2f} dB")
    print(f"Magnitude std:            {np.std(magnitude_db):.2f} dB")
    print(f"Dynamic range:            {np.ptp(magnitude_db):.2f} dB")

    print("\nSTRONGEST SUBCARRIER")
    print(f"Index:                    {subcarriers[strongest]}")
    print(f"Frequency:                {frequencies[strongest]:.2f} Hz")
    print(f"Magnitude:                {magnitude_db[strongest]:.2f} dB")

    print("\nWEAKEST SUBCARRIER")
    print(f"Index:                    {subcarriers[weakest]}")
    print(f"Frequency:                {frequencies[weakest]:.2f} Hz")
    print(f"Magnitude:                {magnitude_db[weakest]:.2f} dB")

    print("\nPER-SUBCARRIER CHANNEL PROFILE")
    print("-" * 76)

    print(
        f"{'SC':>5} {'Frequency':>12} {'Magnitude':>12} "
        f"{'Phase':>12} {'Mag CV':>12} {'Phase Std':>12}"
    )

    for i in range(len(channel)):

        print(
            f"{subcarriers[i]:>5} "
            f"{frequencies[i]:>12.2f} "
            f"{magnitude_db[i]:>12.2f} "
            f"{phase_deg[i]:>12.2f} "
            f"{consistency_data['magnitude_cv'][i]:>12.4f} "
            f"{consistency_data['phase_std_deg'][i]:>12.2f}"
        )

    return magnitude, magnitude_db


# ============================================================
# TAP PROFILE ANALYSIS
# ============================================================

def analyze_taps(channel_data):

    taps = channel_data["taps"]

    tap_energy = np.abs(taps) ** 2
    total_energy = np.sum(tap_energy)

    if total_energy <= EPSILON:
        raise ValueError("TAP profile has zero energy.")

    normalized_energy = tap_energy / total_energy

    sorted_energy = np.sort(normalized_energy)[::-1]

    peak_index = int(np.argmax(tap_energy))
    peak_magnitude = float(np.abs(taps[peak_index]))

    cumulative_energy = np.cumsum(sorted_energy)

    def components_for_energy(threshold):
        return int(np.searchsorted(cumulative_energy, threshold) + 1)

    print("\n" + "=" * 76)
    print("CHANNEL TAP PROFILE")
    print("=" * 76)

    print(f"Number of TAP samples:    {len(taps)}")
    print(f"Peak TAP index:           {peak_index}")
    print(f"Peak TAP magnitude:       {peak_magnitude:.8f}")
    print(f"Total TAP energy:         {total_energy:.8e}")

    print("\nENERGY CONCENTRATION")
    print("-" * 76)

    print(f"Energy in strongest TAP:  {sorted_energy[0] * 100:.2f}%")
    print(f"Energy in strongest 5:    {np.sum(sorted_energy[:5]) * 100:.2f}%")
    print(f"Energy in strongest 10:   {np.sum(sorted_energy[:10]) * 100:.2f}%")

    print(
        f"TAPs containing 50% energy: "
        f"{components_for_energy(0.50)}"
    )

    print(
        f"TAPs containing 90% energy: "
        f"{components_for_energy(0.90)}"
    )

    print(
        f"TAPs containing 99% energy: "
        f"{components_for_energy(0.99)}"
    )

    print("\nSTRONGEST TAP SAMPLES")
    print("-" * 76)

    strongest_indices = np.argsort(tap_energy)[::-1][:15]

    print(f"{'Index':>10} {'Magnitude':>16} {'Energy (%)':>16}")

    for index in strongest_indices:

        print(
            f"{index:>10} "
            f"{np.abs(taps[index]):>16.8f} "
            f"{normalized_energy[index] * 100:>16.4f}"
        )

    print(
        "\nNote: TAP indices describe the stored interpolated "
        "profile, not independently verified physical paths."
    )


# ============================================================
# TAP / FREQUENCY RESPONSE CONSISTENCY
# ============================================================

def analyze_tap_frequency_consistency(channel_data):

    taps = channel_data["taps"]
    channel = channel_data["channel_response"]
    subcarriers = channel_data["active_subcarriers"]

    fft_response = np.fft.fft(taps)

    # Convert signed OFDM subcarrier indices to FFT array indices.
    fft_indices = subcarriers.astype(int) % len(taps)

    reconstructed = fft_response[fft_indices]

    # Fit a complex scalar to account for possible normalization.
    denominator = np.vdot(reconstructed, reconstructed)

    if abs(denominator) <= EPSILON:
        raise ValueError("Invalid FFT reconstruction from TAPs.")

    scale = np.vdot(reconstructed, channel) / denominator

    aligned_reconstruction = scale * reconstructed

    relative_error = (
        np.linalg.norm(channel - aligned_reconstruction)
        / max(np.linalg.norm(channel), EPSILON)
    )

    correlation = abs(
        np.vdot(channel, aligned_reconstruction)
    ) / (
        np.linalg.norm(channel)
        * np.linalg.norm(aligned_reconstruction)
        + EPSILON
    )

    print("\n" + "=" * 76)
    print("TAP / FREQUENCY RESPONSE CONSISTENCY")
    print("=" * 76)

    print(f"Complex scale factor:     {scale}")
    print(f"Relative reconstruction error: {relative_error:.8f}")
    print(f"Normalized correlation:  {correlation:.8f}")

    print(
        "\nThis comparison checks whether the stored TAP profile "
        "reproduces the saved channel response under the assumed "
        "IFFT/FFT convention."
    )


# ============================================================
# CAPTURE AND TEMPORAL CONSISTENCY
# ============================================================

def analyze_consistency(channel_data, consistency_data):

    differences = channel_data["capture_differences"]
    gain_ratios = channel_data["capture_gain_ratios"]
    timing_offsets = channel_data["capture_timing_offsets_samples"]

    temporal_coherence = consistency_data["temporal_coherence"]
    phase_std = consistency_data["phase_std_deg"]
    magnitude_cv = consistency_data["magnitude_cv"]

    print("\n" + "=" * 76)
    print("CAPTURE AND TEMPORAL CONSISTENCY")
    print("=" * 76)

    print(f"Channel marked stable:    {bool(channel_data['stable'])}")

    print("\nCROSS-CAPTURE METRICS")

    for i in range(len(differences)):

        print(
            f"Capture {i + 1}: "
            f"difference={differences[i]:.4f}, "
            f"gain ratio={gain_ratios[i]:.4f}, "
            f"timing offset={timing_offsets[i]:.3f} samples"
        )

    print("\nPER-SUBCARRIER STABILITY")

    print(f"Mean magnitude CV:        {np.mean(magnitude_cv):.6f}")
    print(f"Maximum magnitude CV:     {np.max(magnitude_cv):.6f}")
    print(f"Mean phase std:           {np.mean(phase_std):.4f} deg")
    print(f"Maximum phase std:        {np.max(phase_std):.4f} deg")
    print(f"Mean temporal coherence:  {np.mean(temporal_coherence):.6f}")
    print(f"Minimum temporal coherence: {np.min(temporal_coherence):.6f}")


# ============================================================
# ZF VS MMSE ANALYSIS
# ============================================================

def analyze_equalizers(channel):

    magnitude = np.abs(channel)

    # ZF equalizer gain.
    zf_gain = 1.0 / np.maximum(magnitude, EPSILON)

    print("\n" + "=" * 76)
    print("ZERO-FORCING VS MMSE")
    print("=" * 76)

    print("\nZERO-FORCING GAIN")

    print(f"Maximum gain:             {np.max(zf_gain):.6f}")
    print(f"Mean gain:                {np.mean(zf_gain):.6f}")
    print(f"Maximum gain in dB:       {20 * np.log10(np.max(zf_gain)):.2f} dB")

    print("\nMMSE GAIN UNDER DIFFERENT NOISE SCENARIOS")
    print("-" * 76)

    print(
        f"{'Noise variance':>18} "
        f"{'Max gain':>14} "
        f"{'Mean gain':>14} "
        f"{'Weak SC gain':>16}"
    )

    weakest = int(np.argmin(magnitude))

    for noise_db in NOISE_LEVELS_DB:

        noise_variance = 10 ** (noise_db / 10)

        mmse_gain = (
            magnitude
            / (
                magnitude**2
                + noise_variance / SYMBOL_ENERGY
            )
        )

        print(
            f"{noise_db:>15} dB "
            f"{np.max(mmse_gain):>14.6f} "
            f"{np.mean(mmse_gain):>14.6f} "
            f"{mmse_gain[weakest]:>16.6f}"
        )

    print("\nIMPORTANT")

    print(
        "The noise values are hypothetical scenarios. "
        "They are not measured noise estimates from the real recording."
    )


# ============================================================
# OUTPUT FILE INVENTORY
# ============================================================

def inspect_csv_files():

    print("\n" + "=" * 76)
    print("EXISTING CSV FILES")
    print("=" * 76)

    filenames = [
        "channel_taps.csv",
        "channel_taps_interpolated.csv",
        "channel_frequency_response.csv",
        "capture_metrics.csv",
        "cross_capture_metrics.csv",
    ]

    for filename in filenames:

        path = OUTPUT_DIR / filename

        if not path.exists():
            continue

        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            reader = csv.reader(file)

            header = next(reader, [])
            rows = sum(1 for _ in reader)

        print(f"\n{filename}")
        print(f"Columns: {header}")
        print(f"Data rows: {rows}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print("ACOUSTIC OFDM - EQUALIZER LAB")
    print("CHANNEL AND TAP ANALYSIS")
    print("=" * 76)

    channel_data, consistency_data = load_data()

    validate_data(channel_data, consistency_data)

    print(f"\nData directory: {OUTPUT_DIR}")

    channel, magnitude_db = analyze_channel(
        channel_data,
        consistency_data,
    )

    analyze_taps(channel_data)

    analyze_tap_frequency_consistency(channel_data)

    analyze_consistency(
        channel_data,
        consistency_data,
    )

    analyze_equalizers(channel)

    inspect_csv_files()

    print("\n" + "=" * 76)
    print("ANALYSIS COMPLETED")
    print("=" * 76)
    print("No original channel files modified.")
    print("No new audio recordings generated.")
    print("No receiver files modified.")


if __name__ == "__main__":
    main()
