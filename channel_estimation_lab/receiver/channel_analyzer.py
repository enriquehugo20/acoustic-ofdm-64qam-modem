
import numpy as np
import matplotlib.pyplot as plt

from config import (
    SAMPLE_RATE,
    FFT_SIZE,
    CARRIER_FREQ,
    ACTIVE_SUBCARRIER_INDICES,
    TAP_THRESHOLD_DB,
    OUTPUT_DIR,
)


def analyze_channel(channel_response, impulse_response):

    """
    Analyze the measured frequency response and
    the interpolated, band-limited impulse response.

    IMPORTANT:
    The impulse response is interpolated.
    Its 512 samples are NOT 512 independently
    resolvable physical channel taps.
    """

    channel_response = np.asarray(
        channel_response,
        dtype=np.complex128,
    )

    impulse_response = np.asarray(
        impulse_response,
        dtype=np.complex128,
    )

    subcarriers = np.asarray(
        ACTIVE_SUBCARRIER_INDICES,
        dtype=np.float64,
    )

    if channel_response.shape != subcarriers.shape:
        raise ValueError(
            "Channel response dimensions do not match "
            "the active subcarriers."
        )

    if impulse_response.shape != (FFT_SIZE,):
        raise ValueError(
            f"Expected {FFT_SIZE} impulse response samples."
        )

    # --------------------------------------------------------
    # FREQUENCY RESPONSE
    # --------------------------------------------------------

    subcarrier_spacing = SAMPLE_RATE / FFT_SIZE

    acoustic_frequencies = (
        CARRIER_FREQ + subcarriers * subcarrier_spacing
    )

    magnitude = np.abs(channel_response)

    magnitude_db = 20 * np.log10(
        np.maximum(magnitude, 1e-12)
    )

    phase_rad = np.unwrap(
        np.angle(channel_response)
    )

    phase_deg = np.rad2deg(phase_rad)

    # --------------------------------------------------------
    # FREQUENCY RESPONSE STATISTICS
    # --------------------------------------------------------

    mean_magnitude = np.mean(magnitude)

    median_magnitude = np.median(magnitude)

    min_magnitude = np.min(magnitude)

    max_magnitude = np.max(magnitude)

    mean_magnitude_db = 20 * np.log10(
        max(mean_magnitude, 1e-12)
    )

    median_magnitude_db = 20 * np.log10(
        max(median_magnitude, 1e-12)
    )

    min_magnitude_db = np.min(magnitude_db)

    max_magnitude_db = np.max(magnitude_db)

    magnitude_std_db = np.std(magnitude_db)

    # --------------------------------------------------------
    # APPARENT GROUP DELAY FROM PHASE SLOPE
    # --------------------------------------------------------

    frequency_order = np.argsort(
        acoustic_frequencies
    )

    sorted_frequencies = acoustic_frequencies[
        frequency_order
    ]

    sorted_phase = np.unwrap(
        np.angle(channel_response[frequency_order])
    )

    phase_slope, phase_intercept = np.polyfit(
        sorted_frequencies,
        sorted_phase,
        1,
    )

    apparent_delay_seconds = (
        -phase_slope / (2 * np.pi)
    )

    apparent_delay_us = (
        apparent_delay_seconds * 1e6
    )

    fitted_phase = (
        phase_slope * sorted_frequencies
        + phase_intercept
    )

    phase_residual = sorted_phase - fitted_phase

    phase_residual_std_deg = np.rad2deg(
        np.std(phase_residual)
    )

    # --------------------------------------------------------
    # INTERPOLATED IMPULSE RESPONSE
    # --------------------------------------------------------

    impulse_magnitude = np.abs(
        impulse_response
    )

    impulse_power = impulse_magnitude ** 2

    peak_index = np.argmax(
        impulse_magnitude
    )

    peak_amplitude = impulse_magnitude[
        peak_index
    ]

    peak_power = impulse_power[
        peak_index
    ]

    impulse_magnitude_db = 20 * np.log10(
        np.maximum(
            impulse_magnitude,
            1e-12,
        )
    )

    impulse_power_db = 10 * np.log10(
        np.maximum(
            impulse_power,
            1e-24,
        )
    )

    # fftshift places zero delay at the center.
    delay_samples = np.arange(
        FFT_SIZE
    ) - FFT_SIZE // 2

    delay_us = (
        delay_samples / SAMPLE_RATE
    ) * 1e6

    peak_delay_us = delay_us[
        peak_index
    ]

    # --------------------------------------------------------
    # PROFILE AROUND THE MAIN PEAK
    # --------------------------------------------------------

    threshold = (
        peak_amplitude
        * 10 ** (TAP_THRESHOLD_DB / 20)
    )

    significant = (
        impulse_magnitude >= threshold
    )

    significant_indices = np.where(
        significant
    )[0]

    if len(significant_indices) > 0:

        first_significant_delay = delay_us[
            significant_indices[0]
        ]

        last_significant_delay = delay_us[
            significant_indices[-1]
        ]

        significant_sample_count = len(
            significant_indices
        )

    else:

        first_significant_delay = np.nan
        last_significant_delay = np.nan
        significant_sample_count = 0

    # --------------------------------------------------------
    # ENERGY DISTRIBUTION
    # --------------------------------------------------------

    total_energy = np.sum(
        impulse_power
    )

    peak_energy_percentage = (
        peak_power / total_energy * 100
        if total_energy > 0
        else 0.0
    )

    # Energy in a window around the main peak.
    window_us = 250.0

    around_peak = (
        np.abs(delay_us - peak_delay_us)
        <= window_us
    )

    energy_around_peak = np.sum(
        impulse_power[around_peak]
    )

    energy_around_peak_percentage = (
        energy_around_peak / total_energy * 100
        if total_energy > 0
        else 0.0
    )

    # --------------------------------------------------------
    # FREQUENCY RESPONSE EXTREMES
    # --------------------------------------------------------

    weakest_index = np.argmin(
        magnitude
    )

    strongest_index = np.argmax(
        magnitude
    )

    # --------------------------------------------------------
    # PRINT DIAGNOSTICS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ACOUSTIC CHANNEL ANALYSIS")
    print("=" * 65)

    print("\nSYSTEM CONFIGURATION")
    print("-" * 65)

    print(f"Sample rate:                 {SAMPLE_RATE} Hz")
    print(f"FFT size:                    {FFT_SIZE}")
    print(f"Subcarrier spacing:          {subcarrier_spacing:.3f} Hz")
    print(f"Carrier frequency:           {CARRIER_FREQ} Hz")
    print(f"Active subcarriers:           {len(subcarriers)}")
    print(f"Impulse response samples:     {len(impulse_response)}")
    print(f"Impulse sample interval:      {1e6 / SAMPLE_RATE:.3f} us")

    print("\nFREQUENCY RESPONSE STATISTICS")
    print("-" * 65)

    print(f"Mean |H[k]|:                 {mean_magnitude:.6e}")
    print(f"Median |H[k]|:               {median_magnitude:.6e}")
    print(f"Minimum |H[k]|:              {min_magnitude:.6e}")
    print(f"Maximum |H[k]|:              {max_magnitude:.6e}")

    print(f"Mean magnitude:              {mean_magnitude_db:.2f} dB")
    print(f"Median magnitude:            {median_magnitude_db:.2f} dB")
    print(f"Minimum magnitude:           {min_magnitude_db:.2f} dB")
    print(f"Maximum magnitude:           {max_magnitude_db:.2f} dB")
    print(f"Magnitude standard deviation:{magnitude_std_db:8.2f} dB")

    print("\nSTRONGEST SUBCARRIER")
    print("-" * 65)

    print(
        f"Subcarrier:                  "
        f"{int(subcarriers[strongest_index])}"
    )

    print(
        f"Acoustic frequency:          "
        f"{acoustic_frequencies[strongest_index]:.2f} Hz"
    )

    print(
        f"Magnitude:                   "
        f"{magnitude_db[strongest_index]:.2f} dB"
    )

    print("\nWEAKEST SUBCARRIER")
    print("-" * 65)

    print(
        f"Subcarrier:                  "
        f"{int(subcarriers[weakest_index])}"
    )

    print(
        f"Acoustic frequency:          "
        f"{acoustic_frequencies[weakest_index]:.2f} Hz"
    )

    print(
        f"Magnitude:                   "
        f"{magnitude_db[weakest_index]:.2f} dB"
    )

    print("\nPHASE ANALYSIS")
    print("-" * 65)

    print(
        f"Unwrapped phase range:       "
        f"{np.min(phase_deg):.2f} to "
        f"{np.max(phase_deg):.2f} degrees"
    )

    print(
        f"Apparent group delay:        "
        f"{apparent_delay_us:.3f} us"
    )

    print(
        f"Phase fit residual std:      "
        f"{phase_residual_std_deg:.2f} degrees"
    )

    print(
        "Note: apparent delay includes possible "
        "synchronization timing offset."
    )

    print("\nINTERPOLATED IMPULSE RESPONSE")
    print("-" * 65)

    print(f"Peak sample index:            {peak_index}")
    print(f"Peak delay:                   {peak_delay_us:.3f} us")
    print(f"Peak amplitude:               {peak_amplitude:.6e}")
    print(f"Peak power:                   {peak_power:.6e}")

    print(
        f"Threshold:                    "
        f"{TAP_THRESHOLD_DB} dB relative to peak"
    )

    print(
        f"First significant delay:      "
        f"{first_significant_delay:.3f} us"
    )

    print(
        f"Last significant delay:       "
        f"{last_significant_delay:.3f} us"
    )

    print(
        f"Significant profile samples:  "
        f"{significant_sample_count}"
    )

    print("\nENERGY DISTRIBUTION")
    print("-" * 65)

    print(
        f"Total profile energy:         "
        f"{total_energy:.6e}"
    )

    print(
        f"Peak sample energy:           "
        f"{peak_energy_percentage:.2f}%"
    )

    print(
        f"Energy within +/- {window_us:.0f} us:    "
        f"{energy_around_peak_percentage:.2f}%"
    )

    print("\nACTIVE SUBCARRIER DETAILS")
    print("-" * 65)

    print(
        f"{'Subcarrier':>10} "
        f"{'Frequency (Hz)':>16} "
        f"{'Magnitude (dB)':>16} "
        f"{'Phase (deg)':>14}"
    )

    print("-" * 65)

    for i in frequency_order:

        print(
            f"{int(subcarriers[i]):10d} "
            f"{acoustic_frequencies[i]:16.2f} "
            f"{magnitude_db[i]:16.2f} "
            f"{phase_deg[i]:14.2f}"
        )

    print("\n" + "=" * 65)

    print(
        "IMPORTANT: The impulse response is band-limited "
        "and interpolated."
    )

    print(
        "Its samples must not be interpreted as "
        "independent physical channel taps."
    )

    print(
        "RMS delay spread is intentionally not reported."
    )

    print("=" * 65)

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 16),
    )

    # 1. Impulse response magnitude

    axes[0].plot(
        delay_us,
        impulse_magnitude_db,
    )

    axes[0].axvline(
        peak_delay_us,
        linestyle="--",
        label="Main peak",
    )

    axes[0].axhline(
        TAP_THRESHOLD_DB
        + impulse_magnitude_db[peak_index],
        linestyle=":",
        label=f"{TAP_THRESHOLD_DB} dB threshold",
    )

    axes[0].set_title(
        "Band-Limited Channel Impulse Response"
    )

    axes[0].set_xlabel(
        "Relative delay (microseconds)"
    )

    axes[0].set_ylabel(
        "Magnitude (dB)"
    )

    axes[0].grid(True)
    axes[0].legend()

    # 2. Power delay profile

    axes[1].plot(
        delay_us,
        impulse_power_db,
    )

    axes[1].set_title(
        "Band-Limited Channel Power Delay Profile"
    )

    axes[1].set_xlabel(
        "Relative delay (microseconds)"
    )

    axes[1].set_ylabel(
        "Power (dB)"
    )

    axes[1].grid(True)

    # 3. Frequency response magnitude

    axes[2].plot(
        acoustic_frequencies[frequency_order],
        magnitude_db[frequency_order],
        marker="o",
    )

    axes[2].set_title(
        "Estimated Channel Frequency Response - Magnitude"
    )

    axes[2].set_xlabel(
        "Acoustic frequency (Hz)"
    )

    axes[2].set_ylabel(
        "Magnitude (dB)"
    )

    axes[2].grid(True)

    # 4. Frequency response phase

    axes[3].plot(
        acoustic_frequencies[frequency_order],
        np.rad2deg(sorted_phase),
        marker="o",
    )

    axes[3].set_title(
        "Estimated Channel Frequency Response - Phase"
    )

    axes[3].set_xlabel(
        "Acoustic frequency (Hz)"
    )

    axes[3].set_ylabel(
        "Unwrapped phase (degrees)"
    )

    axes[3].grid(True)

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "channel_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    print(
        f"\nGraph saved: {output_path}"
    )

    plt.show()
