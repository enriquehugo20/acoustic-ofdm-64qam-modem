
# ============================================================
# CRC GENERATOR
# CRC-32 IEEE error detection
# ============================================================

import zlib


def calculate_crc(data: bytes) -> int:
    """
    Calculate CRC-32 IEEE checksum.
    """

    return zlib.crc32(data) & 0xFFFFFFFF


def append_crc(data: bytes) -> bytes:
    """
    Append a 32-bit CRC to the payload.

    Format:
        [PAYLOAD][CRC-32]
    """

    checksum = calculate_crc(data)

    crc_bytes = checksum.to_bytes(
        4,
        byteorder="big"
    )

    return data + crc_bytes
