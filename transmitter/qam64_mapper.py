import numpy as np

# Gray-coded labels mapped to amplitude levels.
GRAY_TO_LEVEL = {
    0b000: -7,
    0b001: -5,
    0b011: -3,
    0b010: -1,
    0b110:  1,
    0b111:  3,
    0b101:  5,
    0b100:  7,
}

# Average energy of the unnormalized square 64-QAM constellation.
NORMALIZATION = np.sqrt(42)


def qam64_map(bits: np.ndarray) -> np.ndarray:
    """
    Convert groups of six binary bits into normalized 64-QAM symbols.

    Bit order:
        First 3 bits  -> In-phase (I)
        Last 3 bits   -> Quadrature (Q)
    """
    raw_bits = np.asarray(bits)

    if raw_bits.ndim != 1:
        raise ValueError("Input bits must be a one-dimensional array.")

    if not np.all(np.isin(raw_bits, [0, 1])):
        raise ValueError("Input must contain only binary values 0 and 1.")

    if len(raw_bits) % 6 != 0:
        raise ValueError("64-QAM input length must be divisible by 6.")

    bits = raw_bits.astype(np.uint8, copy=False)
    groups = bits.reshape(-1, 6)

    symbols = np.empty(len(groups), dtype=np.complex64)

    for index, group in enumerate(groups):
        i_label = int("".join(map(str, group[:3])), 2)
        q_label = int("".join(map(str, group[3:])), 2)

        i_value = GRAY_TO_LEVEL[i_label]
        q_value = GRAY_TO_LEVEL[q_label]

        symbols[index] = complex(i_value, q_value) / NORMALIZATION

    return symbols