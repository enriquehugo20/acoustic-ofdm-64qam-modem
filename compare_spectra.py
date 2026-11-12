
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch

from config import (
    SAMPLE_RATE,
    ACOUSTIC_PREAMBLE_SAMPLES,
    OFDM_SYMBOL_SAMPLES,
    TRAINING_SYMBOLS,
)

FILES = {
    "Original": ("transmission.wav", 0),
    "Recorded": ("5G.wav", 97936),
}

OFDM_SYMBOL_COUNT = TRAINING_SYMBOLS + 2


def load_audio(path):
    fs, x = wavfile.read(path)

    if x.ndim == 2:
        x = x.astype(np.float64).mean(axis=1)
    else:
        x = x.astype(np.float64)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x /= max(abs(info.min), abs(info.max))

    if fs != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate in {path}")

    return x


def extract(path, start):
    x = load_audio(path)

    preamble = x[
        start:start + ACOUSTIC_PREAMBLE_SAMPLES
    ]

    ofdm_start = start + ACOUSTIC_PREAMBLE_SAMPLES

    ofdm = x[
        ofdm_start:
        ofdm_start + OFDM_SYMBOL_COUNT * OFDM_SYMBOL_SAMPLES
    ]

    if len(preamble) != ACOUSTIC_PREAMBLE_SAMPLES:
        raise ValueError(f"Invalid preamble in {path}")

    if len(ofdm) != OFDM_SYMBOL_COUNT * OFDM_SYMBOL_SAMPLES:
        raise ValueError(f"Invalid OFDM section in {path}")

    return preamble, ofdm


signals = {}

for name, (path, start) in FILES.items():
    signals[name] = extract(path, start)

fig, axes = plt.subplots(2, 1, figsize=(12, 9))

for name, (preamble, ofdm) in signals.items():

    # PSD of preamble
    fp, pp = welch(
        preamble,
        fs=SAMPLE_RATE,
        nperseg=4096,
        scaling="density"
    )

    # PSD of OFDM
    fo, po = welch(
        ofdm,
        fs=SAMPLE_RATE,
        nperseg=2048,
        scaling="density"
    )

    axes[0].plot(
        fp,
        10*np.log10(pp + 1e-15),
        label=name
    )

    axes[1].plot(
        fo,
        10*np.log10(po + 1e-15),
        label=name
    )

axes[0].set_title("Acoustic preamble - Power spectral density")
axes[1].set_title("OFDM - Power spectral density")

for ax in axes:
    ax.set_xlim(0, SAMPLE_RATE/2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.grid()
    ax.legend()

plt.tight_layout()
plt.show()
