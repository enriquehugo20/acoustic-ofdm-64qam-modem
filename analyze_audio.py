
import numpy as np
from scipy.io import wavfile

from config import (
    SAMPLE_RATE,
    ACOUSTIC_PREAMBLE_SAMPLES,
    OFDM_SYMBOL_SAMPLES,
    TRAINING_SYMBOLS,
)

ORIGINAL_FILE = "transmission.wav"
RECORDED_FILE = "5G.wav"

# Detected position from your latest real recording.
RECORDED_PREAMBLE_START = 97936

# Short "hola" transmission:
# 2 training + 1 header + 1 payload.
OFDM_SYMBOL_COUNT = TRAINING_SYMBOLS + 2


def load_audio(filepath):
    sample_rate, audio = wavfile.read(filepath)

    original_dtype = audio.dtype

    if audio.ndim == 2:
        audio = audio.astype(np.float64).mean(axis=1)
    else:
        audio = audio.astype(np.float64)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        scale = max(abs(info.min), abs(info.max))
        audio /= scale

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{filepath}: expected {SAMPLE_RATE} Hz, got {sample_rate} Hz"
        )

    return audio


def extract_segments(audio, start, name):
    preamble_start = start
    ofdm_start = start + ACOUSTIC_PREAMBLE_SAMPLES
    ofdm_end = ofdm_start + OFDM_SYMBOL_COUNT * OFDM_SYMBOL_SAMPLES

    if ofdm_end > len(audio):
        raise ValueError(
            f"{name}: audio is too short for the requested segments."
        )

    preamble = audio[
        preamble_start:
        preamble_start + ACOUSTIC_PREAMBLE_SAMPLES
    ]

    ofdm = audio[ofdm_start:ofdm_end]

    return preamble, ofdm


def measure(name, signal):
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    rms_dbfs = 20 * np.log10(max(rms, 1e-12))

    print(f"\n{name}")
    print(f"Samples: {len(signal)}")
    print(f"RMS: {rms:.6f}")
    print(f"Peak: {peak:.6f}")
    print(f"RMS: {rms_dbfs:.2f} dBFS")

    return rms


def compare(name, original_rms, recorded_rms):
    gain_db = 20 * np.log10(
        max(recorded_rms, 1e-12) /
        max(original_rms, 1e-12)
    )

    print(f"{name}: {gain_db:+.2f} dB")


# Load both files.
original = load_audio(ORIGINAL_FILE)
recorded = load_audio(RECORDED_FILE)

print("===== FILE INFORMATION =====")
print(f"Original samples: {len(original)}")
print(f"Recorded samples: {len(recorded)}")

# Extract original transmission.
original_preamble, original_ofdm = extract_segments(
    original, 0, "ORIGINAL"
)

# Extract actual microphone recording.
recorded_preamble, recorded_ofdm = extract_segments(
    recorded, RECORDED_PREAMBLE_START, "RECORDED"
)

# Measure original.
print("\n========== ORIGINAL: transmission.wav ==========")
op = measure("Preamble", original_preamble)
oo = measure("OFDM", original_ofdm)

original_difference = 20 * np.log10(op / oo)

print(f"\nOriginal preamble/OFDM difference: {original_difference:.2f} dB")

# Measure recording.
print("\n========== RECORDED: 5G.wav ==========")
rp = measure("Preamble", recorded_preamble)
ro = measure("OFDM", recorded_ofdm)

recorded_difference = 20 * np.log10(rp / ro)

print(f"\nRecorded preamble/OFDM difference: {recorded_difference:.2f} dB")

# Compare the same sections between files.
print("\n========== CHANNEL LEVEL CHANGE ==========")
compare("Preamble recording gain", op, rp)
compare("OFDM recording gain", oo, ro)

print("\n========== FINAL COMPARISON ==========")
print(f"Original difference: {original_difference:.2f} dB")
print(f"Recorded difference: {recorded_difference:.2f} dB")
print(
    "Change in relative balance:",
    f"{recorded_difference - original_difference:+.2f} dB"
)
