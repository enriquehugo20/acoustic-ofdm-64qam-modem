
import numpy as np

from config import ROLLOFF, SAMPLES_PER_SYMBOL, RRC_SPAN
from transmitter.rrc_filter import root_raised_cosine


def apply_matched_filter(baseband: np.ndarray) -> np.ndarray:
    """
    Apply the RRC matched filter to the received complex baseband signal.
    """

    taps = root_raised_cosine(
        rolloff=ROLLOFF,
        samples_per_symbol=SAMPLES_PER_SYMBOL,
        span=RRC_SPAN,
    )

    return np.convolve(
        baseband,
        taps,
        mode="full",
    )
