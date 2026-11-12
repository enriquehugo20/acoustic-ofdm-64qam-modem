import numpy as np

from scipy.io import wavfile

from config import (
    SAMPLE_RATE,
    CARRIER_FREQ,
    FFT_SIZE,
    OFDM_SYMBOL_SAMPLES,
    AUDIO_SCALE,
    SILENCE_BEFORE,
    SILENCE_AFTER,
    OUTPUT_DIR,
)

from transmitter.training_signal import (
    generate_training_waveform,
)


def complex_to_audio(baseband):

    """
    Convert complex OFDM baseband into real passband audio.
    """

    n = np.arange(len(baseband))

    carrier = np.exp(
        1j * 2 * np.pi
        * CARRIER_FREQ
        * n / SAMPLE_RATE
    )

    audio = np.real(baseband * carrier)

    audio *= AUDIO_SCALE

    peak = np.max(np.abs(audio))

    if peak > 1.0:
        audio *= 0.95 / peak

    return audio.astype(np.float32)


def generate_audio():

    """
    Generate the training-only acoustic transmission.
    """

    training = generate_training_waveform()

    audio = complex_to_audio(training)

    silence_before = np.zeros(
        int(SILENCE_BEFORE * SAMPLE_RATE),
        dtype=np.float32,
    )

    silence_after = np.zeros(
        int(SILENCE_AFTER * SAMPLE_RATE),
        dtype=np.float32,
    )

    waveform = np.concatenate([
        silence_before,
        audio,
        silence_after,
    ])

    output_path = (
        OUTPUT_DIR / "training_transmission.wav"
    )

    wavfile.write(
        output_path,
        SAMPLE_RATE,
        waveform,
    )

    print("\nTRAINING AUDIO GENERATED")
    print("------------------------")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Carrier: {CARRIER_FREQ} Hz")
    print(f"FFT size: {FFT_SIZE}")
    print(
        f"OFDM symbol samples: {OFDM_SYMBOL_SAMPLES}"
    )
    print(
        f"Training symbols: {len(training) // OFDM_SYMBOL_SAMPLES}"
    )
    print(
        f"Duration: {len(waveform) / SAMPLE_RATE:.3f} s"
    )
    print(f"Output: {output_path}")

    return output_path