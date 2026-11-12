
# ============================================================
# TEXT ENCODER
# Converts text into a binary sequence
# ============================================================

import numpy as np


def text_to_bytes(message: str) -> bytes:
    """
    Convert a text message into UTF-8 bytes.

    Input:
        message: Python string

    Output:
        encoded_message: UTF-8 byte sequence
    """

    return message.encode("utf-8")


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Convert bytes into a binary NumPy array.

    Each byte is represented using 8 bits,
    most significant bit first.

    Input:
        data: byte sequence

    Output:
        bits: NumPy array containing 0s and 1s
    """

    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8)
    )


def encode_text(message: str) -> np.ndarray:
    """
    Complete text encoding process.

    Text -> UTF-8 bytes -> binary sequence
    """

    encoded_bytes = text_to_bytes(message)

    bits = bytes_to_bits(encoded_bytes)

    return bits
