
import numpy as np

from config import SAMPLE_RATE


POSTAMBLE_DURATION = 0.25
POSTAMBLE_TONE_OFFSET = 3000.0
POSTAMBLE_FADE_DURATION = 0.01


def generate_acoustic_postamble() -> np.ndarray:
    """
    Generate a deterministic acoustic cosine postamble.

    The complex baseband tone is shifted by the existing
    carrier during complex_to_audio(), producing a 6 kHz
    acoustic tone with the current 3 kHz carrier.
    """

    sample_count = int(
        POSTAMBLE_DURATION * SAMPLE_RATE
    )

    sample_indices = np.arange(sample_count)

    # Complex baseband tone.
    tone = np.exp(
        1j * 2 * np.pi
        * POSTAMBLE_TONE_OFFSET
        * sample_indices / SAMPLE_RATE
    )

    # Smooth fade-in and fade-out.
    envelope = np.ones(sample_count)

    fade_samples = int(
        POSTAMBLE_FADE_DURATION * SAMPLE_RATE
    )

    fade_samples = min(
        fade_samples,
        sample_count // 2,
    )

    fade = np.sin(
        np.linspace(0, np.pi / 2, fade_samples)
    ) ** 2

    envelope[:fade_samples] = fade
    envelope[-fade_samples:] = fade[::-1]

    postamble = tone * envelope

    return postamble.astype(np.complex128)
