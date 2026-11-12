
import numpy as np

from transmitter.qam64_mapper import (
    GRAY_TO_LEVEL,
    NORMALIZATION,
)

LEVEL_TO_GRAY = {
    level: label
    for label, level in GRAY_TO_LEVEL.items()
}

AMPLITUDE_LEVELS = np.array(
    [-7, -5, -3, -1, 1, 3, 5, 7],
    dtype=np.float64,
)


def _nearest_level(value: float) -> int:
    """Return the closest valid 64-QAM amplitude level."""
    index = np.argmin(np.abs(AMPLITUDE_LEVELS - value))
    return int(AMPLITUDE_LEVELS[index])


def _label_to_bits(label: int) -> list:
    """Convert a three-bit integer label into a bit list."""
    return [
        (label >> shift) & 1
        for shift in (2, 1, 0)
    ]


def qam64_demap(
    symbols: np.ndarray,
    decision_gain: np.ndarray = None,
) -> np.ndarray:
    """
    Convert normalized 64-QAM symbols into binary bits.

    decision_gain compensates for the effective MMSE output
    attenuation on each data subcarrier.

    It must contain one positive gain per input symbol.
    """
    symbols = np.asarray(
        symbols,
        dtype=np.complex128,
    )

    if symbols.ndim != 1:
        raise ValueError(
            "Input symbols must be a one-dimensional array."
        )

    if not np.all(np.isfinite(symbols)):
        raise ValueError(
            "Symbols contain invalid numerical values."
        )

    if decision_gain is None:
        gains = np.ones(
            symbols.shape,
            dtype=np.float64,
        )
    else:
        gains = np.asarray(
            decision_gain,
            dtype=np.float64,
        )

        if gains.shape != symbols.shape:
            raise ValueError(
                "Decision gain must match the symbol vector shape."
            )

        if (
            not np.all(np.isfinite(gains))
            or np.any(gains <= 0)
        ):
            raise ValueError(
                "Decision gains must be finite and positive."
            )

    bits = []

    for symbol, gain in zip(symbols, gains):
        # Restore the original constellation scale for hard decisions.
        i_value = (
            symbol.real * NORMALIZATION / gain
        )

        q_value = (
            symbol.imag * NORMALIZATION / gain
        )

        i_level = _nearest_level(i_value)
        q_level = _nearest_level(q_value)

        i_label = LEVEL_TO_GRAY[i_level]
        q_label = LEVEL_TO_GRAY[q_level]

        bits.extend(_label_to_bits(i_label))
        bits.extend(_label_to_bits(q_label))

    return np.asarray(bits, dtype=np.uint8)
