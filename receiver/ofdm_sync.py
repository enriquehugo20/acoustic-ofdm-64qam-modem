
import numpy as np

from scipy.signal import correlate, hilbert

from config import (
    SAMPLE_RATE,
    CARRIER_FREQ,
    SYNC_THRESHOLD,
    ACOUSTIC_PREAMBLE_SAMPLES,
    FFT_SIZE,
    CYCLIC_PREFIX_LENGTH,
    OFDM_SYMBOL_SAMPLES,
    TRAINING_SYMBOLS,
)

from transmitter.acoustic_preamble import generate_acoustic_preamble


def _to_baseband(audio: np.ndarray) -> np.ndarray:
    """
    Convert the complete real passband recording into
    complex analytic baseband.
    """
    analytic_signal = hilbert(
        np.asarray(audio, dtype=np.float64)
    )

    sample_indices = np.arange(len(audio))

    carrier = np.exp(
        -1j * 2 * np.pi
        * CARRIER_FREQ
        * sample_indices
        / SAMPLE_RATE
    )

    return analytic_signal * carrier


def generate_acoustic_preamble_reference() -> np.ndarray:
    return generate_acoustic_preamble().astype(
        np.complex128
    )


def detect_ofdm_frame_start(audio: np.ndarray) -> int:
    """
    Detect the acoustic preamble.

    Returns the approximate beginning of the first
    OFDM training symbol.
    """
    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError("Audio must be a non-empty mono signal.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid samples.")

    reference = generate_acoustic_preamble_reference()

    if len(audio) < len(reference):
        raise ValueError("Audio is shorter than the acoustic preamble.")

    baseband = _to_baseband(audio)

    baseband = baseband - np.mean(baseband)
    reference = reference - np.mean(reference)

    correlation = correlate(
        baseband,
        reference,
        mode="valid",
        method="fft",
    )

    reference_energy = np.sum(
        np.abs(reference) ** 2
    )

    cumulative_energy = np.concatenate((
        [0.0],
        np.cumsum(np.abs(baseband) ** 2),
    ))

    window_energy = (
        cumulative_energy[len(reference):]
        - cumulative_energy[:-len(reference)]
    )

    denominator = np.sqrt(
        reference_energy * window_energy
    )

    scores = np.divide(
        np.abs(correlation),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    acoustic_start = int(np.argmax(scores))
    best_score = float(scores[acoustic_start])

    if best_score < SYNC_THRESHOLD:
        raise ValueError(
            "Acoustic preamble not detected. "
            f"Best correlation: {best_score:.4f}"
        )

    frame_start = (
        acoustic_start + ACOUSTIC_PREAMBLE_SAMPLES
    )

    print("\n===== ACOUSTIC SYNCHRONIZATION =====")
    print(f"Acoustic preamble start: {acoustic_start} samples")
    print(
        f"Acoustic start time: "
        f"{acoustic_start / SAMPLE_RATE:.6f} seconds"
    )
    print(f"Correlation: {best_score:.4f}")
    print(f"Coarse OFDM start: {frame_start} samples")

    return frame_start




def refine_ofdm_sync(audio, coarse_start):
    """
    Refine OFDM synchronization using:
    1. Cyclic-prefix correlation.
    2. Repeated training-symbol correlation.

    Preserve the acoustic preamble boundary.
    """

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim != 1 or not np.all(np.isfinite(audio)):
        raise ValueError("Invalid audio signal.")

    baseband = _to_baseband(audio)
    start = int(coarse_start)

    if start < 0:
        raise ValueError("Invalid OFDM synchronization position.")

    if start + 2 * OFDM_SYMBOL_SAMPLES > len(baseband):
        raise ValueError("Not enough samples for OFDM training symbols.")

    # --------------------------------------------------------
    # 1. CFO estimation using cyclic prefix
    # --------------------------------------------------------

    first_cp = baseband[
        start:start + CYCLIC_PREFIX_LENGTH
    ]

    second_cp = baseband[
        start + FFT_SIZE:
        start + FFT_SIZE + CYCLIC_PREFIX_LENGTH
    ]

    cp_correlation = np.vdot(first_cp, second_cp)

    cp_denominator = np.sqrt(
        np.sum(np.abs(first_cp) ** 2)
        * np.sum(np.abs(second_cp) ** 2)
    )

    cp_score = (
        abs(cp_correlation) / cp_denominator
        if cp_denominator > 0 else 0.0
    )

    if cp_score < 0.1:
        raise ValueError("Cyclic prefix correlation is too weak.")

    cp_frequency_offset = (
        np.angle(cp_correlation)
        * SAMPLE_RATE
        / (2 * np.pi * FFT_SIZE)
    )

    # --------------------------------------------------------
    # 2. CFO estimation using repeated training symbols
    # --------------------------------------------------------

    first_training = baseband[
        start + CYCLIC_PREFIX_LENGTH:
        start + OFDM_SYMBOL_SAMPLES
    ]

    second_training = baseband[
        start + OFDM_SYMBOL_SAMPLES + CYCLIC_PREFIX_LENGTH:
        start + 2 * OFDM_SYMBOL_SAMPLES
    ]

    training_correlation = np.vdot(
        first_training,
        second_training,
    )

    training_denominator = np.sqrt(
        np.sum(np.abs(first_training) ** 2)
        * np.sum(np.abs(second_training) ** 2)
    )

    training_score = (
        abs(training_correlation) / training_denominator
        if training_denominator > 0 else 0.0
    )

    training_frequency_offset = None

    if training_score >= 0.2:
        training_frequency_offset = (
            np.angle(training_correlation)
            * SAMPLE_RATE
            / (2 * np.pi * OFDM_SYMBOL_SAMPLES)
        )

    # --------------------------------------------------------
    # 3. Select CFO estimate
    # --------------------------------------------------------

    if training_frequency_offset is not None:
        frequency_offset = training_frequency_offset
        cfo_source = "Repeated training symbols"
    else:
        frequency_offset = cp_frequency_offset
        cfo_source = "Cyclic prefix"

    # --------------------------------------------------------
    # 4. Diagnostics
    # --------------------------------------------------------

    print("\n===== OFDM FINE SYNCHRONIZATION =====")
    print(f"Training start: {start} samples")
    print(f"CP correlation: {cp_score:.4f}")
    print(f"Training correlation: {training_score:.4f}")

    print(
        f"CFO from CP: "
        f"{cp_frequency_offset:.6f} Hz"
    )

    if training_frequency_offset is not None:
        print(
            f"CFO from training: "
            f"{training_frequency_offset:.6f} Hz"
        )
    else:
        print("CFO from training: unavailable")

    print(f"Selected CFO source: {cfo_source}")
    print(f"Selected CFO: {frequency_offset:.6f} Hz")

    return start, float(frequency_offset)



    # --------------------------------------------------------
    # 1. Fine timing using cyclic-prefix correlation
    # --------------------------------------------------------

    for candidate in range(
        first_candidate,
        last_candidate + 1,
    ):
        first = baseband[
            candidate:
            candidate + CYCLIC_PREFIX_LENGTH
        ]

        second = baseband[
            candidate + FFT_SIZE:
            candidate + FFT_SIZE + CYCLIC_PREFIX_LENGTH
        ]

        if len(first) != CYCLIC_PREFIX_LENGTH:
            continue

        if len(second) != CYCLIC_PREFIX_LENGTH:
            continue

        correlation = np.vdot(
            first,
            second,
        )

        denominator = np.sqrt(
            np.sum(np.abs(first) ** 2)
            * np.sum(np.abs(second) ** 2)
        )

        if denominator <= 0:
            continue

        score = abs(correlation) / denominator

        if score > best_score:
            best_score = float(score)
            best_start = candidate
            best_cp_correlation = correlation

    if best_start is None:
        raise ValueError("Fine OFDM synchronization failed.")

        # --------------------------------------------------------
    # Prefer acoustic boundary when CP correlation is equivalent
    # --------------------------------------------------------

    coarse_candidate = int(coarse_start)

    if first_candidate <= coarse_candidate <= last_candidate:

        first = baseband[
            coarse_candidate:
            coarse_candidate + CYCLIC_PREFIX_LENGTH
        ]

        second = baseband[
            coarse_candidate + FFT_SIZE:
            coarse_candidate + FFT_SIZE + CYCLIC_PREFIX_LENGTH
        ]

        coarse_correlation = np.vdot(first, second)

        denominator = np.sqrt(
            np.sum(np.abs(first) ** 2)
            * np.sum(np.abs(second) ** 2)
        )

        if denominator > 0:
            coarse_score = (
                abs(coarse_correlation) / denominator
            )

            # If the acoustic boundary is almost as good,
            # prefer it to avoid unnecessary sample shifts.
            if coarse_score >= best_score - 0.01:
                best_start = coarse_candidate
                best_score = float(coarse_score)
                best_cp_correlation = coarse_correlation

    # CFO estimated from cyclic-prefix repetition.
    cp_frequency_offset = (
        np.angle(best_cp_correlation)
        * SAMPLE_RATE
        / (2 * np.pi * FFT_SIZE)
    )

    # --------------------------------------------------------
    # 2. Frequency estimation using repeated training symbols
    # --------------------------------------------------------

    frequency_offset = cp_frequency_offset
    repeated_score = 0.0

    if (
        TRAINING_SYMBOLS >= 2
        and best_start + 2 * OFDM_SYMBOL_SAMPLES <= len(audio)
    ):
        first_training = baseband[
            best_start + CYCLIC_PREFIX_LENGTH:
            best_start + OFDM_SYMBOL_SAMPLES
        ]

        second_training = baseband[
            best_start + OFDM_SYMBOL_SAMPLES + CYCLIC_PREFIX_LENGTH:
            best_start + 2 * OFDM_SYMBOL_SAMPLES
        ]

        repeated_correlation = np.vdot(
            first_training,
            second_training,
        )

        repeated_denominator = np.sqrt(
            np.sum(np.abs(first_training) ** 2)
            * np.sum(np.abs(second_training) ** 2)
        )

        if repeated_denominator > 0:
            repeated_score = (
                abs(repeated_correlation)
                / repeated_denominator
            )

        if repeated_score > 0.1:
            frequency_offset = (
                np.angle(repeated_correlation)
                * SAMPLE_RATE
                / (2 * np.pi * OFDM_SYMBOL_SAMPLES)
            )

    print("\n===== OFDM FINE SYNCHRONIZATION =====")
    print(f"Refined training start: {best_start} samples")
    print(f"CP correlation: {best_score:.4f}")
    print(
        f"CFO from cyclic prefix: "
        f"{cp_frequency_offset:.3f} Hz"
    )
    print(
        f"Repeated training correlation: "
        f"{repeated_score:.4f}"
    )
    print(
        f"Estimated frequency offset: "
        f"{frequency_offset:.3f} Hz"
    )

    return best_start, float(frequency_offset)
