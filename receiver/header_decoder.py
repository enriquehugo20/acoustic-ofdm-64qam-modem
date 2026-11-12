import zlib
import numpy as np

from config import BITS_PER_SYMBOL
from transmitter.frame_header import HEADER_SYMBOLS
from receiver.qam64_demapper import qam64_demap


def decode_header(header_symbols: np.ndarray) -> int:
    """
    Decode the frame header and return the payload packet length.
    """

    if len(header_symbols) != HEADER_SYMBOLS:
        raise ValueError(
            f"Expected {HEADER_SYMBOLS} header symbols."
        )

    bits = qam64_demap(header_symbols)

    # The header contains exactly 64 information bits.
    header_bits = bits[:64]

    header_bytes = np.packbits(header_bits).tobytes()

    length_bytes = header_bytes[:4]

    received_crc = int.from_bytes(
        header_bytes[4:8],
        byteorder="big",
    )

    calculated_crc = zlib.crc32(length_bytes) & 0xFFFFFFFF

    if received_crc != calculated_crc:
        raise ValueError("Header CRC verification failed.")

    packet_length_bytes = int.from_bytes(
        length_bytes,
        byteorder="big",
    )

    if packet_length_bytes <= 0:
        raise ValueError("Invalid packet length in header.")

    print(f"Header CRC: SUCCESS")
    print(f"Packet length: {packet_length_bytes} bytes")

    return packet_length_bytes