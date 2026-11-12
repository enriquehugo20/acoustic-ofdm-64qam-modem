
import numpy as np

from config import (
    FFT_SIZE,
    CYCLIC_PREFIX_LENGTH,
    SAMPLE_RATE,
    CARRIER_FREQ,
    AUDIO_SCALE,
)


def ofdm_modulate(grid: np.ndarray) -> np.ndarray:
    """
    Convert a frequency-domain OFDM grid into a time-domain
    complex signal and append its cyclic prefix.
    """
    grid = np.asarray(grid, dtype=np.complex128)

    if grid.shape != (FFT_SIZE,):
        raise ValueError(f"Expected an FFT grid of {FFT_SIZE} bins.")

    # IFFT with unitary normalization.
    useful_symbol = np.fft.ifft(grid) * np.sqrt(FFT_SIZE)

    # Add cyclic prefix.
    cyclic_prefix = useful_symbol[-CYCLIC_PREFIX_LENGTH:]

    return np.concatenate((cyclic_prefix, useful_symbol))


def complex_to_audio(baseband: np.ndarray) -> np.ndarray:
    """
    Translate complex baseband OFDM into real passband audio.
    """
    baseband = np.asarray(baseband, dtype=np.complex128)

    if baseband.ndim != 1 or len(baseband) == 0:
        raise ValueError("Baseband signal must be non-empty.")

    if not np.all(np.isfinite(baseband)):
        raise ValueError("Invalid baseband signal.")

    sample_indices = np.arange(len(baseband))

    carrier = np.exp(
        1j * 2 * np.pi * CARRIER_FREQ
        * sample_indices / SAMPLE_RATE
    )

    audio = np.real(baseband * carrier) * AUDIO_SCALE

    peak = np.max(np.abs(audio))

    if peak > 1.0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)
