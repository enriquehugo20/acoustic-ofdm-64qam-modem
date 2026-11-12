
import numpy as np

from config import (
    FFT_SIZE,
    ACTIVE_SUBCARRIER_INDICES,
    DATA_SUBCARRIERS,
    PILOT_SUBCARRIERS,
)

from transmitter.pilot_generator import generate_pilot_symbols


def build_resource_grid(
    data_symbols: np.ndarray,
    symbol_index: int = 0,
) -> np.ndarray:
    """
    Map data and pilot symbols into the OFDM frequency grid.

    The returned array contains FFT_SIZE complex frequency bins.
    """
    data_symbols = np.asarray(data_symbols, dtype=np.complex128)

    if data_symbols.ndim != 1:
        raise ValueError("Data symbols must be one-dimensional.")

    if len(data_symbols) != len(DATA_SUBCARRIERS):
        raise ValueError(
            f"Expected {len(DATA_SUBCARRIERS)} data symbols."
        )

    if not np.all(np.isfinite(data_symbols)):
        raise ValueError("Invalid data symbols.")

    grid = np.zeros(FFT_SIZE, dtype=np.complex128)

    # Map data subcarriers.
    data_indices = np.asarray(DATA_SUBCARRIERS) % FFT_SIZE
    grid[data_indices] = data_symbols

    # Map known pilot subcarriers.
    pilot_symbols = generate_pilot_symbols(symbol_index)
    pilot_indices = np.asarray(PILOT_SUBCARRIERS) % FFT_SIZE

    grid[pilot_indices] = pilot_symbols

    return grid


def build_preamble_grid():
    rng = np.random.default_rng(2026)

    phases = rng.integers(
        0, 4,
        size=len(ACTIVE_SUBCARRIER_INDICES)
    )

    symbols = np.exp(1j * np.pi / 2 * phases)

    grid = np.zeros(FFT_SIZE, dtype=np.complex128)

    indices = np.asarray(ACTIVE_SUBCARRIER_INDICES) % FFT_SIZE

    grid[indices] = symbols

    return grid