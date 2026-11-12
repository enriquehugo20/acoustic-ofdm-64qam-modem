
import numpy as np

from config import (
    FFT_SIZE,
    CYCLIC_PREFIX_LENGTH,
    OFDM_SYMBOL_SAMPLES,
    SAMPLE_RATE,
    CARRIER_FREQ,
)


def demodulate_ofdm_symbol(
    audio: np.ndarray,
    start_sample: int,
    frequency_offset_hz: float = 0.0,
) -> np.ndarray:
    """
    Convert one real passband OFDM symbol into
    its complex frequency-domain grid.

    Compensates the estimated carrier frequency offset.
    """
    end_sample = (
        start_sample + OFDM_SYMBOL_SAMPLES
    )

    if start_sample < 0 or end_sample > len(audio):
        raise ValueError("Incomplete OFDM symbol.")

    chunk = np.asarray(
        audio[start_sample:end_sample],
        dtype=np.float64,
    )

    sample_indices = np.arange(
        len(chunk)
    )

    corrected_carrier_frequency = (
        CARRIER_FREQ + frequency_offset_hz
    )

    carrier = np.exp(
        -1j * 2 * np.pi
        * corrected_carrier_frequency
        * sample_indices
        / SAMPLE_RATE
    )

    baseband = 2.0 * chunk * carrier

    # Remove cyclic prefix.
    useful_symbol = baseband[
        CYCLIC_PREFIX_LENGTH:
    ]

    if len(useful_symbol) != FFT_SIZE:
        raise ValueError("Invalid OFDM symbol length.")

    grid = (
        np.fft.fft(useful_symbol)
        / np.sqrt(FFT_SIZE)
    )

    return grid


def demodulate_ofdm_frame(
    audio: np.ndarray,
    frame_start: int,
    symbol_count: int,
    frequency_offset_hz: float = 0.0,
) -> list:
    """
    Demodulate consecutive OFDM symbols.
    """
    grids = []

    for index in range(symbol_count):
        start = (
            frame_start
            + index * OFDM_SYMBOL_SAMPLES
        )

        grids.append(
            demodulate_ofdm_symbol(
                audio,
                start,
                frequency_offset_hz,
            )
        )

    return grids
