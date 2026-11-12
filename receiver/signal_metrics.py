
import numpy as np


def calculate_signal_metrics(audio: np.ndarray) -> dict:
    """
    Calculate basic signal quality metrics from captured audio.
    """

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError("Audio must be a non-empty mono signal.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid samples.")

    # --------------------------------------------------------
    # 1. RMS - Root Mean Square
    # --------------------------------------------------------

    rms = float(np.sqrt(np.mean(audio ** 2)))

    # --------------------------------------------------------
    # 2. Peak amplitude
    # --------------------------------------------------------

    peak = float(np.max(np.abs(audio)))

    # --------------------------------------------------------
    # 3. Peak level in dBFS
    # --------------------------------------------------------

    peak_dbfs = (
        float(20 * np.log10(peak))
        if peak > 0 else float("-inf")
    )

    # --------------------------------------------------------
    # 4. RMS level in dBFS
    # --------------------------------------------------------

    rms_dbfs = (
        float(20 * np.log10(rms))
        if rms > 0 else float("-inf")
    )

    # --------------------------------------------------------
    # 5. Clipping detection
    # --------------------------------------------------------

    clipping_samples = int(np.sum(np.abs(audio) >= 0.999))
    clipping_percentage = (
        clipping_samples / len(audio)
    ) * 100

    # --------------------------------------------------------
    # 6. DC offset
    # --------------------------------------------------------

    dc_offset = float(np.mean(audio))

    # --------------------------------------------------------
    # 7. Crest factor
    # --------------------------------------------------------

    crest_factor = (
        peak / rms if rms > 0 else 0.0
    )

    crest_factor_db = (
        float(20 * np.log10(crest_factor))
        if crest_factor > 0 else float("-inf")
    )

    return {
        "samples": len(audio),
        "rms": rms,
        "rms_dbfs": rms_dbfs,
        "peak": peak,
        "peak_dbfs": peak_dbfs,
        "clipping_samples": clipping_samples,
        "clipping_percentage": clipping_percentage,
        "dc_offset": dc_offset,
        "crest_factor": crest_factor,
        "crest_factor_db": crest_factor_db,
    }


def print_signal_metrics(metrics: dict) -> None:
    """
    Display signal quality metrics.
    """

    print("\n===== SIGNAL QUALITY METRICS =====")

    print(f"Samples: {metrics['samples']}")
    print(f"RMS: {metrics['rms']:.6f}")
    print(f"RMS level: {metrics['rms_dbfs']:.2f} dBFS")

    print(f"Peak amplitude: {metrics['peak']:.6f}")
    print(f"Peak level: {metrics['peak_dbfs']:.2f} dBFS")

    print(
        f"Clipping samples: {metrics['clipping_samples']} "
        f"({metrics['clipping_percentage']:.4f}%)"
    )

    print(f"DC offset: {metrics['dc_offset']:.6f}")

    print(f"Crest factor: {metrics['crest_factor']:.4f}")
    print(f"Crest factor: {metrics['crest_factor_db']:.2f} dB")
