
import numpy as np
import zlib

from transmitter.acoustic_postamble import generate_acoustic_postamble

from config import (
    DATA_SUBCARRIERS,
    DATA_BITS_PER_OFDM_SYMBOL,
    TRAINING_SYMBOLS,
    ACOUSTIC_PREAMBLE_SAMPLES,
    SAMPLE_RATE,
)

from transmitter.crc import append_crc
from transmitter.qam64_mapper import qam64_map
from transmitter.resource_grid import build_resource_grid
from transmitter.preamble import generate_preamble
from transmitter.acoustic_preamble import generate_acoustic_preamble
from transmitter.ofdm_modulator import (
    ofdm_modulate,
    complex_to_audio,
)
from transmitter.audio_output import write_wav


def _build_header(packet_length: int) -> bytes:
    length_bytes = packet_length.to_bytes(
        4,
        byteorder="big",
    )

    checksum = zlib.crc32(length_bytes) & 0xFFFFFFFF

    return (
        length_bytes
        + checksum.to_bytes(4, byteorder="big")
    )


def _encode_ofdm_data(data_bits: np.ndarray) -> list:
    bits_per_block = DATA_BITS_PER_OFDM_SYMBOL

    padding = (-len(data_bits)) % bits_per_block

    if padding:
        data_bits = np.concatenate((
            data_bits,
            np.zeros(padding, dtype=np.uint8),
        ))

    blocks = data_bits.reshape(-1, bits_per_block)

    return [
        qam64_map(block)
        for block in blocks
    ]


def transmit(
    message: str,
    filepath: str = "transmission.wav",
):
    if not message:
        raise ValueError("Message cannot be empty.")

    payload = message.encode("utf-8")
    packet = append_crc(payload)
    header = _build_header(len(packet))

    header_bits = np.unpackbits(
        np.frombuffer(header, dtype=np.uint8)
    )

    header_padding = (
        DATA_BITS_PER_OFDM_SYMBOL - len(header_bits)
    )

    if header_padding < 0:
        raise ValueError("Header exceeds one OFDM symbol.")

    header_bits = np.concatenate((
        header_bits,
        np.zeros(header_padding, dtype=np.uint8),
    ))

    header_qam = qam64_map(header_bits)

    header_grid = build_resource_grid(
        header_qam,
        symbol_index=0,
    )

    payload_bits = np.unpackbits(
        np.frombuffer(packet, dtype=np.uint8)
    )

    payload_blocks = _encode_ofdm_data(payload_bits)

    # --------------------------------------------------------
    # OFDM FRAME
    # --------------------------------------------------------

    training_symbols = [
        generate_preamble()
        for _ in range(TRAINING_SYMBOLS)
    ]

    ofdm_symbols = (
        training_symbols
        + [ofdm_modulate(header_grid)]
    )

    for index, qam_symbols in enumerate(
        payload_blocks,
        start=1,
    ):
        grid = build_resource_grid(
            qam_symbols,
            symbol_index=index,
        )

        ofdm_symbols.append(
            ofdm_modulate(grid)
        )

        # --------------------------------------------------------
    # ACOUSTIC FRAME
    # --------------------------------------------------------

    acoustic_preamble = generate_acoustic_preamble()
    ofdm_frame = np.concatenate(ofdm_symbols)
    acoustic_postamble = generate_acoustic_postamble()

    # Concatenate preamble + OFDM + postamble.
    baseband = np.concatenate((
        acoustic_preamble,
        ofdm_frame,
        acoustic_postamble,
    ))

    # Convert the complete frame to audio.
    audio = complex_to_audio(baseband)

    # Save the WAV file.
    write_wav(audio, filepath)

    # --------------------------------------------------------
    # TRANSMISSION INFORMATION
    # --------------------------------------------------------

    print("\n===== ACOUSTIC OFDM TRANSMITTER =====")
    print(f"WAV generated: {filepath}")
    print(f"Payload bytes: {len(payload)}")
    print(f"Packet bytes: {len(packet)}")

    print(
        f"Acoustic preamble: "
        f"{len(acoustic_preamble)} samples"
    )

    print(
        f"Acoustic preamble duration: "
        f"{len(acoustic_preamble) / SAMPLE_RATE:.3f} seconds"
    )

    print(f"OFDM training symbols: {TRAINING_SYMBOLS}")
    print("Header OFDM symbols: 1")
    print(f"Payload OFDM symbols: {len(payload_blocks)}")
    print(f"Total OFDM symbols: {len(ofdm_symbols)}")

    print(f"Postamble: {len(acoustic_postamble)} samples")
    print(f"Data subcarriers per symbol: {len(DATA_SUBCARRIERS)}")
    print(f"Total duration: {len(audio) / SAMPLE_RATE:.3f} seconds")

    return filepath