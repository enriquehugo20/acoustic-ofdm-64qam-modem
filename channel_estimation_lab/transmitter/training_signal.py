import numpy as np

from config import (
    FFT_SIZE,
    CYCLIC_PREFIX_LENGTH,
    ACTIVE_SUBCARRIER_INDICES,
    TRAINING_SEED,
    TRAINING_SYMBOLS,
    TRAINING_BLOCK_REPETITIONS,
    SAMPLE_RATE,
)


def build_training_grid():

    """
    Generate the known OFDM training resource grid.

    Uses the same random seed and active subcarrier
    configuration as the original modem.
    """

    rng = np.random.default_rng(TRAINING_SEED)

    phases = rng.integers(
        0,
        4,
        size=len(ACTIVE_SUBCARRIER_INDICES),
    )

    qpsk_symbols = np.exp(
        1j * np.pi / 2 * phases
    )

    grid = np.zeros(
        FFT_SIZE,
        dtype=np.complex128,
    )

    for subcarrier, symbol in zip(
        ACTIVE_SUBCARRIER_INDICES,
        qpsk_symbols,
    ):

        grid[subcarrier % FFT_SIZE] = symbol

    return grid


def ofdm_modulate(grid):

    """
    Convert frequency-domain OFDM grid into time domain.
    """

    useful_symbol = (
        np.fft.ifft(grid) * np.sqrt(FFT_SIZE)
    )

    cyclic_prefix = useful_symbol[
        -CYCLIC_PREFIX_LENGTH:
    ]

    return np.concatenate([
        cyclic_prefix,
        useful_symbol,
    ])


def generate_training_symbols():

    """
    Generate the two original identical training symbols.
    """

    grid = build_training_grid()

    symbol = ofdm_modulate(grid)

    return np.tile(
        symbol,
        TRAINING_SYMBOLS,
    )


def generate_training_waveform():

    """
    Repeat the original two-symbol training block
    to obtain more channel observations.
    """

    block = generate_training_symbols()

    return np.tile(
        block,
        TRAINING_BLOCK_REPETITIONS,
    )


def get_known_training_grid():

    """
    Return the exact frequency-domain reference
    used by the receiver.
    """

    return build_training_grid()