
import numpy as np

from config import (
    PILOT_SUBCARRIERS,
    ACTIVE_SUBCARRIER_INDICES,
)

from transmitter.pilot_generator import generate_pilot_symbols


def estimate_channel(
    received_pilots: np.ndarray,
    symbol_index: int,
) -> np.ndarray:
    """
    Estimate the channel response using known pilot symbols.

    Linear interpolation is applied across active subcarriers.
    """
    received_pilots = np.asarray(
        received_pilots,
        dtype=np.complex128,
    )

    expected_pilots = generate_pilot_symbols(symbol_index)

    if received_pilots.shape != expected_pilots.shape:
        raise ValueError("Invalid pilot vector.")

    if not np.all(np.isfinite(received_pilots)):
        raise ValueError("Invalid received pilot values.")

    pilot_indices = np.asarray(PILOT_SUBCARRIERS)

    # Least-squares channel estimates at pilot positions.
    pilot_channel = received_pilots / expected_pilots

    active_indices = np.asarray(
        ACTIVE_SUBCARRIER_INDICES
    )

    # Interpolate real and imaginary components separately.
    channel_real = np.interp(
        active_indices,
        pilot_indices,
        pilot_channel.real,
    )

    channel_imag = np.interp(
        active_indices,
        pilot_indices,
        pilot_channel.imag,
    )

    return channel_real + 1j * channel_imag
