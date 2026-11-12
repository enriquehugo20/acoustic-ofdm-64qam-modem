
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
from config import (
    SAMPLE_RATE,
    ACOUSTIC_PREAMBLE_SAMPLES,
    OFDM_SYMBOL_SAMPLES,
)
import matplotlib.pyplot as plt


# Usage:
# python diagnose_channel.py RECORDED_START
#
# Example:
# python diagnose_channel.py 141481

if len(sys.argv) != 2:
    print("Usage: python diagnose_channel.py RECORDED_START")
    sys.exit(1)

RECORDED_START = int(sys.argv[1])


def load_audio(path):
    fs, audio = wavfile.read(path)
    original_dtype = audio.dtype

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        audio = audio.astype(np.float64)
        audio /= max(abs(info.min), abs(info.max))
    else:
        audio = audio.astype(np.float64)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if fs != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {fs}")

    return audio


def metrics(x):
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    crest = peak / rms if rms > 0 else 0

    rms_db = 20 * np.log10(rms + 1e-15)
    peak_db = 20 * np.log10(peak + 1e-15)

    return rms, peak, crest, rms_db, peak_db


def print_metrics(name, x):
    rms, peak, crest, rms_db, peak_db = metrics(x)

    print(f"\n{name}")
    print(f"Samples:       {len(x)}")
    print(f"RMS:           {rms:.8f}")
    print(f"Peak:          {peak:.8f}")
    print(f"Crest factor:  {crest:.3f}")
    print(f"RMS dBFS:      {rms_db:.2f} dB")
    print(f"Peak dBFS:     {peak_db:.2f} dB")


def db_difference(a, b):
    return 20 * np.log10((b + 1e-15) / (a + 1e-15))


# --------------------------------------------------
# 1. LOAD FILES
# --------------------------------------------------

tx = load_audio("transmission.wav")
rx = load_audio("5G.wav")

print("\n========== FILE INFORMATION ==========")
print(f"Sample rate:       {SAMPLE_RATE} Hz")
print(f"Original samples:  {len(tx)}")
print(f"Recorded samples:  {len(rx)}")
print(f"Original duration: {len(tx) / SAMPLE_RATE:.3f} s")
print(f"Recorded duration: {len(rx) / SAMPLE_RATE:.3f} s")
print(f"Recorded start:    {RECORDED_START}")

# --------------------------------------------------
# 2. EXTRACT MATCHING FRAME SEGMENTS
# --------------------------------------------------

preamble_len = ACOUSTIC_PREAMBLE_SAMPLES
tx_ofdm_start = preamble_len
rx_ofdm_start = RECORDED_START + preamble_len

tx_preamble = tx[:preamble_len]
rx_preamble = rx[RECORDED_START:RECORDED_START + preamble_len]

if len(tx_preamble) != preamble_len:
    raise ValueError("Original preamble incomplete")

if len(rx_preamble) != preamble_len:
    raise ValueError("Recorded preamble incomplete")

available_tx_symbols = (
    len(tx) - tx_ofdm_start
) // OFDM_SYMBOL_SAMPLES

available_rx_symbols = (
    len(rx) - rx_ofdm_start
) // OFDM_SYMBOL_SAMPLES

symbol_count = min(available_tx_symbols, available_rx_symbols)

if symbol_count < 1:
    raise ValueError("No complete OFDM symbols available")

tx_ofdm = tx[
    tx_ofdm_start:
    tx_ofdm_start + symbol_count * OFDM_SYMBOL_SAMPLES
]

rx_ofdm = rx[
    rx_ofdm_start:
    rx_ofdm_start + symbol_count * OFDM_SYMBOL_SAMPLES
]

print(f"\nComplete OFDM symbols compared: {symbol_count}")

# --------------------------------------------------
# 3. PREAMBLE COMPARISON
# --------------------------------------------------

print("\n========== PREAMBLE COMPARISON ==========")

print_metrics("Original preamble", tx_preamble)
print_metrics("Recorded preamble", rx_preamble)

tx_pre_rms = metrics(tx_preamble)[0]
rx_pre_rms = metrics(rx_preamble)[0]

print(f"\nPreamble level change: {db_difference(tx_pre_rms, rx_pre_rms):.2f} dB")

# --------------------------------------------------
# 4. COMPLETE OFDM COMPARISON
# --------------------------------------------------

print("\n========== COMPLETE OFDM COMPARISON ==========")

print_metrics("Original OFDM", tx_ofdm)
print_metrics("Recorded OFDM", rx_ofdm)

tx_ofdm_rms = metrics(tx_ofdm)[0]
rx_ofdm_rms = metrics(rx_ofdm)[0]

print(f"\nOFDM level change: {db_difference(tx_ofdm_rms, rx_ofdm_rms):.2f} dB")

print("\n========== RELATIVE LEVEL CHANGE ==========")

original_balance = 20 * np.log10(
    (tx_pre_rms + 1e-15) / (tx_ofdm_rms + 1e-15)
)

recorded_balance = 20 * np.log10(
    (rx_pre_rms + 1e-15) / (rx_ofdm_rms + 1e-15)
)

print(f"Original preamble/OFDM: {original_balance:.2f} dB")
print(f"Recorded preamble/OFDM: {recorded_balance:.2f} dB")
print(f"Balance change:         {recorded_balance - original_balance:.2f} dB")

# --------------------------------------------------
# 5. SYMBOL-BY-SYMBOL COMPARISON
# --------------------------------------------------

print("\n========== OFDM SYMBOL ANALYSIS ==========")
print("Symbol | TX RMS dBFS | RX RMS dBFS | Gain dB | RX Peak")

gains = []

for i in range(symbol_count):
    start = i * OFDM_SYMBOL_SAMPLES
    end = start + OFDM_SYMBOL_SAMPLES

    tx_symbol = tx_ofdm[start:end]
    rx_symbol = rx_ofdm[start:end]

    tx_m = metrics(tx_symbol)
    rx_m = metrics(rx_symbol)

    gain = db_difference(tx_m[0], rx_m[0])
    gains.append(gain)

    print(
        f"{i + 1:6d} | "
        f"{tx_m[3]:11.2f} | "
        f"{rx_m[3]:11.2f} | "
        f"{gain:7.2f} | "
        f"{rx_m[1]:.6f}"
    )

print(f"\nMean OFDM symbol gain: {np.mean(gains):.2f} dB")
print(f"Minimum symbol gain:   {np.min(gains):.2f} dB")
print(f"Maximum symbol gain:   {np.max(gains):.2f} dB")
print(f"Gain standard dev.:    {np.std(gains):.2f} dB")

# --------------------------------------------------
# 6. SPECTRAL COMPARISON
# --------------------------------------------------

print("\n========== SPECTRAL ANALYSIS ==========")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

segments = [
    ("Preamble", tx_preamble, rx_preamble),
    ("OFDM", tx_ofdm, rx_ofdm),
]

for column, (name, original, recorded) in enumerate(segments):
    for label, signal in [
        ("Original", original),
        ("Recorded", recorded),
    ]:
        frequencies, psd = welch(
            signal,
            fs=SAMPLE_RATE,
            nperseg=min(2048, len(signal)),
            scaling="density",
        )

        psd_db = 10 * np.log10(psd + 1e-15)

        axes[0, column].plot(
            frequencies, psd_db, label=label
        )

        axes[1, column].plot(
            frequencies,
            psd_db - np.max(psd_db),
            label=label,
        )

    axes[0, column].set_title(f"{name} - Absolute PSD")
    axes[1, column].set_title(f"{name} - Normalized PSD")

    for row in range(2):
        axes[row, column].set_xlabel("Frequency (Hz)")
        axes[row, column].set_ylabel("Power (dB/Hz)")
        axes[row, column].grid(True)
        axes[row, column].legend()

plt.tight_layout()
plt.savefig("channel_diagnostic_spectrum.png", dpi=150)

print("Spectrum graph: channel_diagnostic_spectrum.png")

# --------------------------------------------------
# 7. FINAL WARNINGS
# --------------------------------------------------

print("\n========== DIAGNOSTIC NOTES ==========")
print("These measurements compare fixed-length segments.")
print("They do not independently prove AGC, frequency response,")
print("timing drift, or another specific channel impairment.")
print("Verify frame alignment before interpreting the results.")
