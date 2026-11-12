
# ============================================================
# TEXT DECODER
# Converts recovered UTF-8 bytes into text
# ============================================================


def decode_text(payload: bytes) -> str:
    """
    Decode a recovered UTF-8 payload.

    Input:
        payload: original message bytes

    Output:
        Decoded Python string
    """

    return payload.decode("utf-8")
