import numpy as np

from transmitter.preamble import generate_preamble


def detect_preamble(
    received_symbols: np.ndarray,
    threshold: float = 0.80,
) -> int:
    """
    Detect the known BPSK preamble using normalized correlation.

    Returns the symbol index where the preamble begins.
    """

    received_symbols = np.asarray(
        received_symbols,
        dtype=np.complex128,
    )

    preamble = generate_preamble()

    if received_symbols.ndim != 1:
        raise ValueError("Received symbols must be one-dimensional.")

    if len(received_symbols) < len(preamble):
        raise ValueError("Not enough symbols to detect the preamble.")

    if not np.all(np.isfinite(received_symbols)):
        raise ValueError("Received symbols contain invalid values.")

    reference_energy = np.linalg.norm(preamble)

    best_score = -1.0
    best_position = -1

    for position in range(
        len(received_symbols) - len(preamble) + 1
    ):
        window = received_symbols[
            position:position + len(preamble)
        ]

        received_energy = np.linalg.norm(window)

        if received_energy == 0:
            continue

        correlation = np.vdot(preamble, window)

        score = (
            abs(correlation)
            / (reference_energy * received_energy)
        )

        if score > best_score:
            best_score = score
            best_position = position

    if best_position < 0 or best_score < threshold:
        raise ValueError(
            f"Preamble not detected. Best correlation: {best_score:.3f}"
        )

    print(
        f"Preamble detected at symbol {best_position} "
        f"(correlation: {best_score:.4f})"
    )

    return best_position