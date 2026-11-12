
# ============================================================
# ACOUSTIC 5G-INSPIRED MODEM
# Global configuration - OFDM PHY
# ============================================================

# ============================================================
# AUDIO PARAMETERS
# ============================================================

SAMPLE_RATE = 48000
CARRIER_FREQ = 6000


# ============================================================
# OFDM PARAMETERS
# ============================================================

FFT_SIZE = 512
CYCLIC_PREFIX_LENGTH = 64

SUBCARRIER_SPACING = SAMPLE_RATE / FFT_SIZE

# Number of active subcarriers, excluding DC
ACTIVE_SUBCARRIERS = 48

# Subcarrier indices relative to DC
ACTIVE_SUBCARRIER_INDICES = (
    list(range(-24, 0)) +
    list(range(1, 25))
)


# ============================================================
# PILOT CONFIGURATION
# ============================================================

PILOT_SUBCARRIERS = [
    -21, -15, -9, -3,
      3,   9, 15, 21
]

PILOT_COUNT = len(PILOT_SUBCARRIERS)

DATA_SUBCARRIERS = [
    index
    for index in ACTIVE_SUBCARRIER_INDICES
    if index not in PILOT_SUBCARRIERS
]

DATA_SUBCARRIER_COUNT = len(DATA_SUBCARRIERS)


# ============================================================
# DIGITAL MODULATION
# ============================================================

MODULATION = "64QAM"

BITS_PER_SYMBOL = 6


# ============================================================
# FRAME PARAMETERS
# ============================================================

CRC_BYTES = 4
TRAINING_SYMBOLS = 2


# ============================================================
# ERROR CORRECTION
# ============================================================

FEC_ENABLED = False
FEC_TYPE = None


# ============================================================
# VALIDATION
# ============================================================

assert SAMPLE_RATE > 0
assert 0 < CARRIER_FREQ < SAMPLE_RATE / 2

assert FFT_SIZE > 0
assert FFT_SIZE & (FFT_SIZE - 1) == 0

assert 0 < CYCLIC_PREFIX_LENGTH < FFT_SIZE

assert ACTIVE_SUBCARRIERS == len(ACTIVE_SUBCARRIER_INDICES)
assert PILOT_COUNT == 8
assert DATA_SUBCARRIER_COUNT == 40

assert BITS_PER_SYMBOL == 6
assert len(set(PILOT_SUBCARRIERS)) == PILOT_COUNT

assert all(
    index in ACTIVE_SUBCARRIER_INDICES
    for index in PILOT_SUBCARRIERS
)

assert all(
    index != 0
    for index in ACTIVE_SUBCARRIER_INDICES
)

# ============================================================
# OFDM DERIVED PARAMETERS
# ============================================================


OFDM_SYMBOL_SAMPLES = FFT_SIZE + CYCLIC_PREFIX_LENGTH

DATA_BITS_PER_OFDM_SYMBOL = (
    DATA_SUBCARRIER_COUNT * BITS_PER_SYMBOL
)

AUDIO_SCALE = 0.5

SYNC_THRESHOLD = 0.55

# Acoustic synchronization preamble
ACOUSTIC_PREAMBLE_LENGTH = 307
ACOUSTIC_PREAMBLE_SAMPLES_PER_CHIP = 80

ACOUSTIC_PREAMBLE_SAMPLES = (
    ACOUSTIC_PREAMBLE_LENGTH
    * ACOUSTIC_PREAMBLE_SAMPLES_PER_CHIP
)

ACOUSTIC_PREAMBLE_DURATION = (
    ACOUSTIC_PREAMBLE_SAMPLES / SAMPLE_RATE
)