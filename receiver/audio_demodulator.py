
import numpy as np

from config import (
    SAMPLE_RATE,
    CARRIER_FREQ,
    SAMPLES_PER_SYMBOL,
    RRC_SPAN,
)

from receiver.rrc_matched_filter import apply_matched_filter


def demodulate_audio(audio: np.ndarray) -> np.ndarray:
    """
    Coherently downconvert the real passband audio into complex I/Q,
    apply the matched RRC filter, and recover the QAM symbols.

    This initial receiver assumes:
    - No acoustic channel between transmitter and receiver.
    - Known carrier frequency and phase.
    - Known sample timing and symbol rate.
    """

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError("Audio must be a non-empty 1D array.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid samples.")

    # 1. Recreate the carrier reference.
    sample_indices = np.arange(len(audio))

    carrier_phase = (
        2 * np.pi * CARRIER_FREQ
        * sample_indices / SAMPLE_RATE
    )

    # 2. Coherent I/Q downconversion.
    complex_carrier = np.exp(-1j * carrier_phase)

    baseband = 2 * audio * complex_carrier

    # 3. Apply the matched RRC filter.
    filtered = apply_matched_filter(baseband)

    # 4. Calculate the combined transmit/receive filter delay.
    filter_length = RRC_SPAN * SAMPLES_PER_SYMBOL + 1
    total_delay = filter_length - 1

    # 5. Sample once per symbol.
    symbol_samples = filtered[
        total_delay::SAMPLES_PER_SYMBOL
    ]

    return symbol_samples.astype(np.complex64)
