
import numpy as np

from config import (
    SAMPLE_RATE,
    CARRIER_FREQ,
    SAMPLES_PER_SYMBOL,
    ROLLOFF,
    RRC_SPAN,
)

from transmitter.rrc_filter import root_raised_cosine


def modulate_to_audio(symbols: np.ndarray) -> np.ndarray:
    """
    Pulse-shape complex 64-QAM symbols and convert them
    into a real-valued passband audio waveform.
    """

    symbols = np.asarray(symbols, dtype=np.complex128)

    if symbols.ndim != 1 or len(symbols) == 0:
        raise ValueError("Symbols must be a non-empty 1D array.")

    if not np.all(np.isfinite(symbols)):
        raise ValueError("Symbols contain invalid values.")

    # 1. Generate the transmit Root Raised Cosine filter.
    taps = root_raised_cosine(
        rolloff=ROLLOFF,
        samples_per_symbol=SAMPLES_PER_SYMBOL,
        span=RRC_SPAN,
    )

    # 2. Upsample: insert zeros between QAM symbols.
    upsampled = np.zeros(
        len(symbols) * SAMPLES_PER_SYMBOL,
        dtype=np.complex128,
    )

    upsampled[::SAMPLES_PER_SYMBOL] = symbols

    # 3. Apply pulse shaping to the I/Q baseband signal.
    baseband = np.convolve(
        upsampled,
        taps,
        mode="full",
    )

    # 4. Generate the passband carrier.
    sample_indices = np.arange(len(baseband))

    carrier_phase = (
        2 * np.pi * CARRIER_FREQ
        * sample_indices / SAMPLE_RATE
    )

    # 5. Convert complex I/Q into real audio.
    audio_signal = (
        baseband.real * np.cos(carrier_phase)
        -
        baseband.imag * np.sin(carrier_phase)
    )

    return audio_signal.astype(np.float32)
