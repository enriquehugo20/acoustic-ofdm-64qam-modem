
import zlib
import numpy as np

from receiver.postamble_detector import detect_postamble

from config import (
    OFDM_SYMBOL_SAMPLES,
    DATA_BITS_PER_OFDM_SYMBOL,
    TRAINING_SYMBOLS,
)

from receiver.audio_input import read_wav
from receiver.ofdm_sync import (
    detect_ofdm_frame_start,
    refine_ofdm_sync,
)
from receiver.ofdm_demodulator import demodulate_ofdm_symbol
from receiver.resource_grid_extractor import extract_resource_grid
from receiver.channel_estimation import estimate_channel
from receiver.equalizer import (
    equalize_data,
    estimate_noise_variance,
)
from receiver.qam64_demapper import qam64_demap
from receiver.crc_checker import verify_crc


def _decode_header(
    data_symbols: np.ndarray,
    decision_gain: np.ndarray,
) -> tuple:
    """
    Decode header information without making a failed
    header CRC an immediate reception failure.
    """
    bits = qam64_demap(
    data_symbols,
    decision_gain=decision_gain,
    )

    header_bytes = np.packbits(
        bits[:64]
    ).tobytes()

    length_bytes = header_bytes[:4]

    received_crc = int.from_bytes(
        header_bytes[4:8],
        byteorder="big",
    )

    calculated_crc = (
        zlib.crc32(length_bytes) & 0xFFFFFFFF
    )

    header_crc_ok = (
        received_crc == calculated_crc
    )

    packet_length = int.from_bytes(
        length_bytes,
        byteorder="big",
    )

    print("\n===== HEADER DIAGNOSTICS =====")
    print(
        f"Header CRC: "
        f"{'SUCCESS' if header_crc_ok else 'FAILED'}"
    )
    print(f"Header packet length: {packet_length} bytes")

    return packet_length, header_crc_ok



def _receive_data_symbol(
    audio: np.ndarray,
    start_sample: int,
    symbol_index: int,
    frequency_offset_hz: float,
) -> np.ndarray:
    """
    Demodulate, estimate the channel and noise,
    and apply MMSE equalization to one OFDM symbol.
    """
    received_grid = demodulate_ofdm_symbol(
        audio,
        start_sample,
        frequency_offset_hz,
    )

    # Extract data and pilot subcarriers.
    received_data, received_pilots = extract_resource_grid(
        received_grid
    )

    # Estimate the channel from known pilot symbols.
    channel = estimate_channel(
        received_pilots,
        symbol_index,
    )

    # Estimate effective noise from inactive subcarriers.
    noise_variance = estimate_noise_variance(
        received_grid
    )

    # Apply MMSE equalization.
    return equalize_data(
    received_data,
    channel,
    noise_variance,
    method="mmse",
    return_gain=True,
    )



def _estimate_payload_symbol_count(
    audio: np.ndarray,
    payload_start: int,
) -> int:
    """
    Estimate the number of payload OFDM symbols
    from the energy of the recorded signal.
    """
    available_samples = len(audio) - payload_start

    symbol_count = (
        available_samples // OFDM_SYMBOL_SAMPLES
    )

    if symbol_count <= 0:
        return 0

    rms_values = []

    for index in range(symbol_count):
        start = (
            payload_start
            + index * OFDM_SYMBOL_SAMPLES
        )

        end = start + OFDM_SYMBOL_SAMPLES

        chunk = audio[start:end]

        rms_values.append(
            np.sqrt(np.mean(np.square(chunk)))
        )

    rms_values = np.asarray(rms_values)

    tail_count = max(
        3,
        int(len(rms_values) * 0.10),
    )

    noise_floor = float(
        np.median(rms_values[-tail_count:])
    )

    threshold = max(
        noise_floor * 3.0,
        0.002,
    )

    active_symbols = np.flatnonzero(
        rms_values > threshold
    )

    if len(active_symbols) == 0:
        return symbol_count

    return int(active_symbols[-1]) + 1


def _find_packet_by_crc(
    recovered_bytes: bytes,
) -> tuple:
    """
    Search packet boundaries using the payload CRC.
    """
    for packet_length in range(
        5,
        len(recovered_bytes) + 1,
    ):
        candidate = recovered_bytes[:packet_length]

        if verify_crc(candidate):
            return candidate, True

    return recovered_bytes, False


def receive(
    filepath: str = "transmission.wav",
) -> str:
    """
    Complete acoustic OFDM receiver.
    """

    audio, _ = read_wav(filepath)

    # --------------------------------------------------------
    # 1. Acoustic synchronization
    # --------------------------------------------------------

    coarse_start = detect_ofdm_frame_start(audio)

    # --------------------------------------------------------
    # 2. Fine timing and frequency synchronization
    # --------------------------------------------------------

    frame_start, frequency_offset_hz = refine_ofdm_sync(
        audio,
        coarse_start,
    )

    # --------------------------------------------------------
    # 3. Header decoding
    # --------------------------------------------------------

    header_start = (
        frame_start
        + TRAINING_SYMBOLS * OFDM_SYMBOL_SAMPLES
    )

    header_data, header_gain = _receive_data_symbol(
    audio,
    header_start,
    symbol_index=0,
    frequency_offset_hz=frequency_offset_hz,
    )

    header_packet_length, header_crc_ok = _decode_header(
    header_data,
    header_gain,
    )

    # --------------------------------------------------------
    # 4. Physical and logical payload boundary
    # --------------------------------------------------------

    payload_start = (
        header_start + OFDM_SYMBOL_SAMPLES
    )

    # Detect the physical end of transmission.
    postamble_start, postamble_score, postamble_end = (
        detect_postamble(
            audio,
            search_start=payload_start,
        )
    )

    if postamble_start is not None:
        print("\n===== PHYSICAL POSTAMBLE =====")
        print(f"Postamble detected: {postamble_start}")
        print(f"Correlation score: {postamble_score:.4f}")

        physical_payload_end = postamble_start

    else:
        print("\nPostamble not detected. Using WAV boundary.")
        physical_payload_end = len(audio)

        # Available samples before the physical end.
    available_samples = max(
        0,
        physical_payload_end - payload_start,
    )

    # The postamble detector may locate the boundary
    # a few samples before or after the exact OFDM boundary.
    estimated_symbols = int(round(
        available_samples / OFDM_SYMBOL_SAMPLES
    ))

    aligned_boundary = (
        estimated_symbols * OFDM_SYMBOL_SAMPLES
    )

    boundary_error = abs(
        available_samples - aligned_boundary
    )

    # Correct only small boundary detection errors.
    if postamble_start is not None and boundary_error <= 4:
        available_symbols = estimated_symbols

        print(
            f"Postamble boundary adjustment: "
            f"{boundary_error} sample(s)"
        )

    else:
        available_symbols = (
            available_samples // OFDM_SYMBOL_SAMPLES
        )

    

    # Calculate the expected payload size from the header.
    if header_crc_ok and header_packet_length > 4:

        payload_bits = header_packet_length * 8

        expected_symbols = (
            payload_bits + DATA_BITS_PER_OFDM_SYMBOL - 1
        ) // DATA_BITS_PER_OFDM_SYMBOL

        payload_symbol_count = min(
            expected_symbols,
            available_symbols,
        )

        print("\n===== PAYLOAD BOUNDARY =====")
        print(f"Expected symbols: {expected_symbols}")
        print(f"Physically available symbols: {available_symbols}")
        print(f"Selected symbols: {payload_symbol_count}")

        if available_symbols < expected_symbols:
            print("WARNING: Incomplete physical payload.")

    else:
        print("Header CRC failed. Using energy-based detection.")

        payload_symbol_count = _estimate_payload_symbol_count(
            audio[:physical_payload_end],
            payload_start,
        )

    if payload_symbol_count <= 0:
        print("No complete payload symbols available.")
        return ""

    print("\n===== PAYLOAD RECEPTION =====")
    print(f"Detected payload symbols: {payload_symbol_count}")

    # --------------------------------------------------------
    # 5. Payload demodulation
    # --------------------------------------------------------

    recovered_bits = []

    for index in range(payload_symbol_count):
        symbol_start = (
            payload_start
            + index * OFDM_SYMBOL_SAMPLES
        )

        equalized_symbols, decision_gain = _receive_data_symbol(
        audio,
        symbol_start,
        symbol_index=index + 1,
        frequency_offset_hz=frequency_offset_hz,
        )

        symbol_bits = qam64_demap(
        equalized_symbols,
        decision_gain=decision_gain,
        )

        recovered_bits.extend(
            symbol_bits.tolist()
        )

    recovered_bits = np.asarray(
        recovered_bits,
        dtype=np.uint8,
    )

    recovered_bytes = np.packbits(
        recovered_bits
    ).tobytes()

    # --------------------------------------------------------
    # 6. Packet boundary and CRC recovery
    # --------------------------------------------------------

    recovered_packet, payload_crc_ok = _find_packet_by_crc(
        recovered_bytes
    )

    if payload_crc_ok:
        payload = recovered_packet[:-4]

        print("Payload CRC: SUCCESS")
        print(
            f"Recovered packet length: "
            f"{len(recovered_packet)} bytes"
        )

    else:
        print("Payload CRC: FAILED")

        # Use the header length only if it is plausible.
        if (
            header_packet_length > 4
            and header_packet_length <= len(recovered_bytes)
        ):
            recovered_packet = recovered_bytes[
                :header_packet_length
            ]

            payload = recovered_packet[:-4]

            print(
                "Using header length for diagnostic recovery."
            )

        else:
            # Best-effort recovery when both CRC and header fail.
            payload = (
                recovered_bytes[:-4]
                if len(recovered_bytes) > 4
                else recovered_bytes
            )

            print(
                "Using all detected payload data "
                "for diagnostic recovery."
            )

    # --------------------------------------------------------
    # 7. Text reconstruction
    # --------------------------------------------------------

    message = payload.decode(
        "utf-8",
        errors="replace",
    )

    print("\n===== OFDM RECEIVER RESULTS =====")
    print(f"Detected frame start: {frame_start} samples")
    print(
        f"Estimated frequency offset: "
        f"{frequency_offset_hz:.3f} Hz"
    )
    print(
        f"Header CRC: "
        f"{'SUCCESS' if header_crc_ok else 'FAILED'}"
    )
    print(
        f"Payload CRC: "
        f"{'SUCCESS' if payload_crc_ok else 'FAILED'}"
    )
    print(f"Payload OFDM symbols: {payload_symbol_count}")
    print(f"Recovered message: {message!r}")

    if not payload_crc_ok:
        print(
            "WARNING: Recovered text may contain "
            "transmission errors."
        )

    return message
