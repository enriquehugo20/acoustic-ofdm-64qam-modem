
import numpy as np
from scipy.signal import correlate, hilbert

from transmitter.acoustic_postamble import generate_acoustic_postamble

from config import (
    SAMPLE_RATE,
    CARRIER_FREQ,
)


def detect_postamble(
    audio,
    search_start=0,
    threshold=0.35,
):
    """
    Detect the independent cosine acoustic postamble
    using normalized complex-baseband correlation.
    """

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    if audio.ndim != 1 or len(audio) == 0:
        return None, 0.0, None

    search_start = max(
        0,
        int(search_start),
    )

    # Convert recording to complex analytic baseband.
    analytic = hilbert(audio)

    sample_indices = np.arange(len(audio))

    carrier = np.exp(
        -1j * 2 * np.pi
        * CARRIER_FREQ
        * sample_indices / SAMPLE_RATE
    )

    baseband = analytic * carrier

    # Generate the expected postamble.
    template = generate_acoustic_postamble()

    template_length = len(template)

    if len(baseband) - search_start < template_length:
        return None, 0.0, None

    signal = baseband[search_start:]

    # Matched-filter correlation.
    correlation = correlate(
        signal,
        template,
        mode="valid",
        method="fft",
    )

    # Sliding-window energy normalization.
    signal_energy = np.abs(signal) ** 2

    cumulative_energy = np.concatenate((
        [0.0],
        np.cumsum(signal_energy),
    ))

    window_energy = (
        cumulative_energy[template_length:]
        - cumulative_energy[:-template_length]
    )

    template_energy = np.sum(
        np.abs(template) ** 2
    )

    denominator = np.sqrt(
        template_energy * window_energy
    )

    scores = np.divide(
        np.abs(correlation),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    if best_score < threshold:
        return None, best_score, None

    start_index = search_start + best_index
    end_index = start_index + template_length

    return start_index, best_score, end_index
