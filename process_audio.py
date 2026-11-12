
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def analyze_audio(audio, label):
    audio = np.asarray(audio, dtype=np.float64)

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    peak_db = 20 * np.log10(max(peak, 1e-12))
    rms_db = 20 * np.log10(max(rms, 1e-12))

    clipping = int(np.sum(np.abs(audio) >= 1.0))

    print(f"\n===== {label} =====")
    print(f"Peak:      {peak:.6f} ({peak_db:.2f} dBFS)")
    print(f"RMS:       {rms:.6f} ({rms_db:.2f} dBFS)")
    print(f"Crest:     {peak / max(rms, 1e-12):.4f}")
    print(f"Clipping:  {clipping} samples")

    return peak, rms


def process_audio(input_path, output_path, target_peak):
    audio, sample_rate = sf.read(
        input_path,
        dtype="float64",
        always_2d=True
    )

    if audio.size == 0:
        raise ValueError("Input audio is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Input contains invalid samples.")

    original_peak, original_rms = analyze_audio(
        audio, "ORIGINAL AUDIO"
    )

    if original_peak == 0:
        raise ValueError("Cannot normalize a silent signal.")

    # Apply one constant gain to the entire waveform.
    gain = target_peak / original_peak
    processed = audio * gain

    if np.max(np.abs(processed)) > 1.0:
        raise ValueError("Processing would cause clipping.")

    processed_peak, processed_rms = analyze_audio(
        processed, "PROCESSED AUDIO"
    )

    print("\n===== PROCESSING PARAMETERS =====")
    print(f"Sample rate:       {sample_rate} Hz")
    print(f"Channels:          {audio.shape[1]}")
    print(f"Target peak:       {target_peak:.4f}")
    print(f"Linear gain:       {gain:.6f}")
    print(f"Gain in dB:        {20 * np.log10(gain):.3f} dB")

    # Verify constant gain before writing.
    error = np.max(np.abs(processed - audio * gain))
    print(f"Gain verification error: {error:.12e}")

    output_path = Path(output_path)

    sf.write(
        str(output_path),
        processed,
        sample_rate,
        subtype="PCM_24"
    )

    # Read the saved WAV again to verify its actual properties.
    saved_audio, saved_rate = sf.read(
        str(output_path),
        dtype="float64",
        always_2d=True
    )

    print("\n===== SAVED FILE VERIFICATION =====")
    print(f"Output:       {output_path.resolve()}")
    print(f"Sample rate:  {saved_rate} Hz")
    print(f"Subtype:      {sf.info(str(output_path)).subtype}")
    print(f"Frames:       {len(saved_audio)}")

    analyze_audio(saved_audio, "SAVED PCM AUDIO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global peak normalization for OFDM audio."
    )

    parser.add_argument(
        "--input",
        default="transmission.wav"
    )

    parser.add_argument(
        "--output",
        default="transmission_processed.wav"
    )

    parser.add_argument(
        "--target-peak",
        type=float,
        default=0.98
    )

    args = parser.parse_args()

    if not 0 < args.target_peak <= 0.98:
        raise ValueError(
            "Target peak must be greater than 0 and <= 0.98."
        )

    process_audio(
        args.input,
        args.output,
        args.target_peak
    )
