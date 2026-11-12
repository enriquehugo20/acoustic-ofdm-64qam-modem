from pathlib import Path

import numpy as np
import soundfile as sf

from config import SAMPLE_RATE


def write_wav(
    signal: np.ndarray,
    filepath: str = "transmission.wav",
) -> Path:
    """Save an audio waveform as a 32-bit floating-point WAV file."""

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 1 or len(signal) == 0:
        raise ValueError("Audio signal must be a non-empty 1D array.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Audio signal contains invalid values.")

    output_path = Path(filepath)
    sf.write(
        output_path,
        signal,
        SAMPLE_RATE,
        subtype="FLOAT",
        format="WAV",
    )

    return output_path