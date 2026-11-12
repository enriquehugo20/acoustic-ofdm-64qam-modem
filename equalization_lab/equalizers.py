import numpy as np

from config import EPSILON


def zero_forcing(received, channel):
    """
    Zero Forcing equalizer.

    received: [..., subcarriers]
    channel:  [subcarriers]
    """

    denominator = np.where(
        np.abs(channel) ** 2 < EPSILON,
        EPSILON + 0j,
        channel,
    )

    return received / denominator


def mmse_equalizer(received, channel, noise_variance):
    """
    MMSE equalizer.

    noise_variance:
        Estimated noise variance per subcarrier.
    """

    numerator = np.conjugate(channel)

    denominator = (
        np.abs(channel) ** 2
        + np.maximum(noise_variance, EPSILON)
    )

    weights = numerator / denominator

    return received * weights