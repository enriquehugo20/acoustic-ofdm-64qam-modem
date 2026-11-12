
# ============================================================
# CRC CHECKER
# Validates the integrity of a received packet
# ============================================================

import zlib


def verify_crc(packet: bytes) -> bool:
    """
    Verify CRC-32 appended to a received packet.

    Input:
        packet: [PAYLOAD][CRC-32]

    Output:
        True: CRC valid
        False: CRC invalid
    """

    if len(packet) < 4:
        return False

    payload = packet[:-4]

    received_crc = int.from_bytes(
        packet[-4:],
        byteorder="big"
    )

    calculated_crc = zlib.crc32(payload) & 0xFFFFFFFF

    return calculated_crc == received_crc


def extract_payload(packet: bytes) -> bytes:
    """
    Extract payload after successful CRC verification.
    """

    if not verify_crc(packet):
        raise ValueError("CRC verification failed.")

    return packet[:-4]
