
import numpy as np

from config import (
    ACOUSTIC_PREAMBLE_LENGTH,
    ACOUSTIC_PREAMBLE_SAMPLES_PER_CHIP,
)


def generate_acoustic_preamble() -> np.ndarray:
    """
    Generate a Zadoff-Chu acoustic synchronization sequence.

    The sequence is generated in complex baseband.
    Each Zadoff-Chu chip is repeated to obtain the
    required audio sampling rate.
    """

    length = ACOUSTIC_PREAMBLE_LENGTH
    root = 1

    if length <= 1 or np.gcd(root, length) != 1:
        raise ValueError("Invalid Zadoff-Chu parameters.")

    sample_indices = np.arange(length)

    # Generate the Zadoff-Chu sequence.
    zc_sequence = np.exp(
        -1j * np.pi * root
        * sample_indices
        * (sample_indices + 1)
        / length
    )

    # Expand each chip into audio samples.
    preamble = np.repeat(
        zc_sequence,
        ACOUSTIC_PREAMBLE_SAMPLES_PER_CHIP,
    )

    return preamble.astype(np.complex128)
