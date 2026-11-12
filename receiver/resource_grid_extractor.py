
import numpy as np

from config import (
    DATA_SUBCARRIERS,
    PILOT_SUBCARRIERS,
    FFT_SIZE,
)


def extract_resource_grid(
    received_grid: np.ndarray,
) -> tuple:
    """
    Extract received data and pilot subcarriers.
    """
    received_grid = np.asarray(
        received_grid,
        dtype=np.complex128,
    )

    if received_grid.shape != (FFT_SIZE,):
        raise ValueError("Invalid received resource grid.")

    data_indices = np.asarray(DATA_SUBCARRIERS) % FFT_SIZE
    pilot_indices = np.asarray(PILOT_SUBCARRIERS) % FFT_SIZE

    data_symbols = received_grid[data_indices]
    pilot_symbols = received_grid[pilot_indices]

    return data_symbols, pilot_symbols
