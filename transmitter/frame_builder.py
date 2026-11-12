

# ============================================================
# FRAME BUILDER
# Prepares the payload for digital modulation
# ============================================================

import numpy as np

from config import BITS_PER_SYMBOL

from transmitter.text_encoder import (
    text_to_bytes,
    bytes_to_bits
)

from transmitter.crc import append_crc


def build_frame(message: str) -> dict:
    """
    Build a transmission packet.

    Text -> UTF-8 -> CRC-32 -> Bits -> QAM padding
    """

    # 1. Convert text into UTF-8 bytes
    payload = text_to_bytes(message)

    # 2. Append CRC-32
    packet = append_crc(payload)

    # 3. Convert packet into bits
    bits = bytes_to_bits(packet)

    # 4. Calculate padding for 64-QAM
    padding_bits = (-len(bits)) % BITS_PER_SYMBOL

    if padding_bits:
        padding = np.zeros(padding_bits, dtype=np.uint8)
        bits = np.concatenate((bits, padding))

    # 5. Return frame and metadata
    return {
        "payload": payload,
        "packet": packet,
        "bits": bits,
        "payload_length": len(payload),
        "packet_length_bytes": len(packet),
        "bits_before_padding": len(packet) * 8,
        "padding_bits": padding_bits,
        "total_bits": len(bits),
        "symbol_count": len(bits) // BITS_PER_SYMBOL
    }
