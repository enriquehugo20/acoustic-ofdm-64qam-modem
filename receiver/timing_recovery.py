
# ============================================================
# TRANSMITTER
# Coordinates the digital transmission chain
# ============================================================

from transmitter.frame_builder import build_frame
from transmitter.qam64_mapper import qam64_map


def transmit_ideal(message: str) -> dict:
    """
    Prepare a message and generate its 64-QAM symbols.

    No audio output or channel simulation is performed.
    """

    # 1. Build packet and prepare bits.
    frame = build_frame(message)

    # 2. Map bits into complex 64-QAM symbols.
    symbols = qam64_map(frame["bits"])

    return {
        "symbols": symbols,
        "metadata": {
            "packet_length_bytes": frame["packet_length_bytes"],
            "payload_length": frame["payload_length"],
            "padding_bits": frame["padding_bits"]
        },
        "original_message": message
    }
