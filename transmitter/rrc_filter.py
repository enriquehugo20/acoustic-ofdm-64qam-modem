import numpy as np


def root_raised_cosine(
    rolloff: float,
    samples_per_symbol: int,
    span: int,
) -> np.ndarray:
    """
    Generate a Root Raised Cosine (RRC) FIR filter.

    The filter is normalized to unit energy.
    """
    if not 0 < rolloff <= 1:
        raise ValueError("Rolloff must be in the interval (0, 1].")

    if samples_per_symbol <= 0 or span <= 0:
        raise ValueError("Filter parameters must be positive.")

    # Odd number of taps gives an integer group delay.
    sample_count = span * samples_per_symbol + 1

    time = (
        np.arange(sample_count) - (sample_count - 1) / 2
    ) / samples_per_symbol

    h = np.zeros(sample_count, dtype=np.float64)

    for index, t in enumerate(time):
        if np.isclose(t, 0.0):
            h[index] = 1 + rolloff * (4 / np.pi - 1)

        elif np.isclose(abs(t), 1 / (4 * rolloff)):
            h[index] = (
                rolloff / np.sqrt(2)
                * (
                    (1 + 2 / np.pi)
                    * np.sin(np.pi / (4 * rolloff))
                    +
                    (1 - 2 / np.pi)
                    * np.cos(np.pi / (4 * rolloff))
                )
            )

        else:
            numerator = (
                np.sin(np.pi * t * (1 - rolloff))
                +
                4 * rolloff * t
                * np.cos(np.pi * t * (1 + rolloff))
            )

            denominator = (
                np.pi * t
                * (1 - (4 * rolloff * t) ** 2)
            )

            h[index] = numerator / denominator

    # Unit-energy normalization.
    h /= np.sqrt(np.sum(h ** 2))

    return h