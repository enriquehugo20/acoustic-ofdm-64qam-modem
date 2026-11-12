from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

LAB_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LAB_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# ORIGINAL OFDM PARAMETERS
# ============================================================

SAMPLE_RATE = 48000

CARRIER_FREQ = 6000

FFT_SIZE = 512

CYCLIC_PREFIX_LENGTH = 64

OFDM_SYMBOL_SAMPLES = FFT_SIZE + CYCLIC_PREFIX_LENGTH


# ============================================================
# SUBCARRIER CONFIGURATION
# ============================================================

ACTIVE_SUBCARRIER_INDICES = (
    list(range(-24, 0))
    + list(range(1, 25))
)

PILOT_SUBCARRIERS = [
    -21, -15, -9, -3,
    3, 9, 15, 21
]

DATA_SUBCARRIERS = [
    k for k in ACTIVE_SUBCARRIER_INDICES
    if k not in PILOT_SUBCARRIERS
]


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

TRAINING_SYMBOLS = 2

TRAINING_SEED = 2026

TRAINING_BLOCK_REPETITIONS = 16

TOTAL_TRAINING_SYMBOLS = (
    TRAINING_SYMBOLS * TRAINING_BLOCK_REPETITIONS
)


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

AUDIO_SCALE = 0.5

SILENCE_BEFORE = 0.5

SILENCE_AFTER = 0.5


# ============================================================
# CHANNEL ESTIMATION
# ============================================================

CHANNEL_TAPS = 32

# Maximum number of taps to estimate.
# Must not exceed the number of active subcarriers.

# Detection threshold for the impulse response.
TAP_THRESHOLD_DB = -20