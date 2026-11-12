
import numpy as np

from config import (
    FFT_SIZE,
    ACTIVE_SUBCARRIER_INDICES,
    DATA_SUBCARRIERS,
)


def estimate_noise_variance(
    received_grid: np.ndarray,
) -> float:
    """
    Estimate effective noise and residual-interference variance
    from inactive OFDM subcarriers.

    This is an effective estimate, not necessarily pure acoustic
    noise. CFO leakage and other residual interference may contribute.
    """
    received_grid = np.asarray(
        received_grid,
        dtype=np.complex128,
    )

    if received_grid.shape != (FFT_SIZE,):
        raise ValueError("Invalid received resource grid.")

    if not np.all(np.isfinite(received_grid)):
        raise ValueError("Invalid received grid values.")

    active_indices = (
        np.asarray(ACTIVE_SUBCARRIER_INDICES, dtype=int) % FFT_SIZE
    )

    inactive_mask = np.ones(FFT_SIZE, dtype=bool)
    inactive_mask[active_indices] = False

    inactive_symbols = received_grid[inactive_mask]

    if inactive_symbols.size == 0:
        raise ValueError("No inactive subcarriers available.")

    inactive_power = np.abs(inactive_symbols) ** 2

    # For complex Gaussian noise, median(power) = mean(power) * ln(2).
    noise_variance = (
        np.median(inactive_power) / np.log(2.0)
    )

    return float(
        max(noise_variance, np.finfo(np.float64).eps)
    )


def _extract_data_channel(
    channel_estimate: np.ndarray,
) -> np.ndarray:
    channel_estimate = np.asarray(
        channel_estimate,
        dtype=np.complex128,
    )

    if channel_estimate.shape != (
        len(ACTIVE_SUBCARRIER_INDICES),
    ):
        raise ValueError("Invalid channel estimate.")

    if not np.all(np.isfinite(channel_estimate)):
        raise ValueError("Invalid channel estimate values.")

    active_positions = {
        subcarrier: position
        for position, subcarrier
        in enumerate(ACTIVE_SUBCARRIER_INDICES)
    }

    return np.asarray(
        [
            channel_estimate[active_positions[index]]
            for index in DATA_SUBCARRIERS
        ],
        dtype=np.complex128,
    )


def equalize_data(
    received_data: np.ndarray,
    channel_estimate: np.ndarray,
    noise_variance: float,
    method: str = "mmse",
    return_gain: bool = False,
):
    """
    Equalize received 64-QAM data symbols.

    MMSE:
        W = conj(H) / (abs(H)**2 + noise_variance)

    ZF:
        W = 1 / H

    When return_gain=True, also return the effective real gain
    applied to the transmitted constellation:

        G = W * H

    The receiver uses this gain to adjust hard-decision thresholds.
    """
    received_data = np.asarray(
        received_data,
        dtype=np.complex128,
    )

    if received_data.shape != (len(DATA_SUBCARRIERS),):
        raise ValueError("Invalid received data vector.")

    if not np.all(np.isfinite(received_data)):
        raise ValueError("Invalid received data values.")

    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("Invalid noise variance.")

    if method not in ("mmse", "zf"):
        raise ValueError("Unsupported equalization method.")

    data_channel = _extract_data_channel(channel_estimate)

    channel_power = np.abs(data_channel) ** 2

    if method == "zf":
        minimum_gain = 1e-8

        if np.any(np.abs(data_channel) < minimum_gain):
            raise ValueError(
                "Channel estimate contains near-zero subcarriers."
            )

        equalized = received_data / data_channel
        effective_gain = np.ones_like(channel_power)

    else:
        # The transmitted normalized 64-QAM constellation has Es = 1.
        denominator = channel_power + noise_variance

        equalized = (
            np.conjugate(data_channel)
            * received_data
            / denominator
        )

        # MMSE output is attenuated by this effective gain.
        effective_gain = (
            channel_power / denominator
        )

    if return_gain:
        return equalized, effective_gain

    return equalized
