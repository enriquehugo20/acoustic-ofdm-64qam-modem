
import numpy as np

from config import PILOT_COUNT


def generate_pilot_symbols(symbol_index: int = 0) -> np.ndarray:
    """
    Generate deterministic BPSK pilot symbols.

    Both transmitter and receiver use the same symbol index
    to reproduce the expected pilot sequence.
    """
    if symbol_index < 0:
        raise ValueError("Symbol index cannot be negative.")

    rng = np.random.default_rng(9000 + symbol_index)

    bits = rng.integers(
        0, 2,
        size=PILOT_COUNT,
        dtype=np.uint8
    )

    return (1.0 - 2.0 * bits).astype(np.complex128)
