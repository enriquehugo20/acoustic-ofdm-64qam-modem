import numpy as np

from config import (
    CHANNEL_FILE,
    TRAINING_SEED,
    TOTAL_SYMBOLS,
)


def load_channel_data():
    if not CHANNEL_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {CHANNEL_FILE}"
        )

    data = np.load(CHANNEL_FILE, allow_pickle=False)

    required = [
        "channel_response",
        "symbol_channel_estimates",
        "active_subcarriers",
        "acoustic_frequencies",
        "stable",
    ]

    missing = [key for key in required if key not in data.files]

    if missing:
        raise ValueError(
            f"Faltan variables en channel_estimate.npz: {missing}"
        )

    estimates = data["symbol_channel_estimates"]

    if estimates.shape != (3, TOTAL_SYMBOLS, 48):
        raise ValueError(
            "Dimensiones inesperadas en symbol_channel_estimates: "
            f"{estimates.shape}"
        )

    if not np.all(np.isfinite(estimates)):
        raise ValueError(
            "La estimación contiene valores no finitos."
        )

    return {
        "channel_response": data["channel_response"],
        "symbol_channel_estimates": estimates,
        "active_subcarriers": data["active_subcarriers"],
        "acoustic_frequencies": data["acoustic_frequencies"],
        "stable": bool(data["stable"]),
    }


def generate_known_training_symbols(number_of_subcarriers=48):
    """
    Reproduce exactamente la secuencia QPSK del laboratorio original.
    """

    rng = np.random.default_rng(TRAINING_SEED)

    phases = rng.integers(
        0,
        4,
        size=number_of_subcarriers,
    )

    symbols = np.exp(
        1j * np.pi / 2 * phases
    )

    return symbols, phases