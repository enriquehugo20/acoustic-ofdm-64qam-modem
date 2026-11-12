
import numpy as np

from config import (
    SAMPLE_RATE,
    CARRIER_FREQ,
    SAMPLES_PER_SYMBOL,
    ROLLOFF,
    RRC_SPAN,
)

from transmitter.preamble import generate_preamble
from transmitter.rrc_filter import root_raised_cosine


def generate_preamble_reference() -> np.ndarray:
    """
    Generate the passband waveform corresponding to the
    known preamble, using the same pulse shaping and carrier
    as the transmitter.
    """

    preamble_symbols = generate_preamble()

    taps = root_raised_cosine(
        rolloff=ROLLOFF,
        samples_per_symbol=SAMPLES_PER_SYMBOL,
        span=RRC_SPAN,
    )

    upsampled = np.zeros(
        len(preamble_symbols) * SAMPLES_PER_SYMBOL,
        dtype=np.complex128,
    )

    upsampled[::SAMPLES_PER_SYMBOL] = preamble_symbols

    baseband = np.convolve(upsampled, taps, mode="full")

    sample_indices = np.arange(len(baseband))

    carrier_phase = (
        2 * np.pi * CARRIER_FREQ
        * sample_indices / SAMPLE_RATE
    )

    reference = (
        baseband.real * np.cos(carrier_phase)
        - baseband.imag * np.sin(carrier_phase)
    )

    return reference.astype(np.float64)


def detect_frame_start(
    audio: np.ndarray,
    threshold: float = 0.65,
) -> int:
    """
    Locate the known preamble in a real passband audio signal.

    Returns the estimated starting sample of the preamble.
    """

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError("Audio must be a non-empty mono signal.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid samples.")

    reference = generate_preamble_reference()

    if len(audio) < len(reference):
        raise ValueError("Audio is shorter than the preamble reference.")

    reference = reference - np.mean(reference)
    reference_energy = np.sum(reference ** 2)

    if reference_energy == 0:
        raise ValueError("Invalid preamble reference.")

    audio_centered = audio - np.mean(audio)

    correlation = np.correlate(
        audio_centered,
        reference,
        mode="valid",
    )

    cumulative_energy = np.concatenate((
        [0.0],
        np.cumsum(audio_centered ** 2),
    ))

    window_energy = (
        cumulative_energy[len(reference):]
        - cumulative_energy[:-len(reference)]
    )

    denominator = np.sqrt(reference_energy * window_energy)

    scores = np.divide(
        np.abs(correlation),
        denominator,
        out=np.zeros_like(correlation),
        where=denominator > 0,
    )

    best_position = int(np.argmax(scores))
    best_score = float(scores[best_position])

    if best_score < threshold:
        raise ValueError(
            f"Preamble not detected in audio. "
            f"Best correlation: {best_score:.4f}"
        )

    print("\n===== TIMING SYNCHRONIZATION =====")
    print(f"Detected frame start: {best_position} samples")
    print(f"Estimated time offset: {best_position / SAMPLE_RATE:.6f} seconds")
    print(f"Correlation score: {best_score:.4f}")

    return best_position
