import zlib
import numpy as np

from config import BITS_PER_SYMBOL
from transmitter.qam64_mapper import qam64_map


HEADER_BYTES = 8
HEADER_BITS = HEADER_BYTES * 8
HEADER_SYMBOLS = (HEADER_BITS + BITS_PER_SYMBOL - 1) // BITS_PER_SYMBOL


def build_header(packet_length_bytes: int) -> np.ndarray:
    """
    Encode packet length and header CRC32 into 64-QAM symbols.

    Header format:
        4 bytes: packet length
        4 bytes: CRC32 of the packet length
    """

    if not 0 < packet_length_bytes <= 0xFFFFFFFF:
        raise ValueError("Packet length must fit in an unsigned 32-bit integer.")

    length_bytes = packet_length_bytes.to_bytes(
        4,
        byteorder="big",
    )

    checksum = zlib.crc32(length_bytes) & 0xFFFFFFFF

    header = (
        length_bytes
        + checksum.to_bytes(4, byteorder="big")
    )

    bits = np.unpackbits(
        np.frombuffer(header, dtype=np.uint8)
    )

    padding_bits = (-len(bits)) % BITS_PER_SYMBOL

    if padding_bits:
        bits = np.concatenate((
            bits,
            np.zeros(padding_bits, dtype=np.uint8),
        ))

    return qam64_map(bits)