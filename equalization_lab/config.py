from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHANNEL_FILE = (
    PROJECT_DIR
    / "channel_estimation_lab"
    / "output"
    / "channel_estimate.npz"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_SEED = 2026
TRAINING_SYMBOLS = 2
BLOCK_REPETITIONS = 16

TOTAL_SYMBOLS = TRAINING_SYMBOLS * BLOCK_REPETITIONS
ESTIMATION_SYMBOLS = 16
VALIDATION_SYMBOLS = 16

EPSILON = 1e-10