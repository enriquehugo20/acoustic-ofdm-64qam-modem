
import numpy as np
import soundfile as sf

from config import SAMPLE_RATE


def read_wav(filepath: str) -> tuple:
    """
    Load WAV audio, convert stereo to mono
    and validate the sampling rate.
    """

    audio, sample_rate = sf.read(
        filepath,
        dtype="float32",
    )

    # Convert stereo to mono.
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Validate audio dimensions.
    if audio.ndim != 1:
        raise ValueError(
            "Unsupported audio format. Expected mono or stereo WAV."
        )

    # Validate sampling rate.
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Invalid sample rate: {sample_rate} Hz. "
            f"Expected {SAMPLE_RATE} Hz."
        )

    print(f"WAV loaded: {filepath}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio samples: {len(audio)}")

    return audio, sample_rate
