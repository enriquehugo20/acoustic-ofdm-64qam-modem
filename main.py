
from transmitter.transmitter import transmit
from receiver.receiver import receive
from receiver.audio_input import read_wav
from receiver.signal_metrics import (
    calculate_signal_metrics,
    print_signal_metrics,
)


def main():
    print("====================================")
    print("       ACOUSTIC OFDM MODEM")
    print("====================================")

    message = input("\nEnter the message to transmit: ").strip()

    if not message:
        print("Message cannot be empty.")
        return

    # ========================================================
    # 1. TRANSMITTER
    # ========================================================

    transmit(message)

    # ========================================================
    # 2. SIGNAL QUALITY METRICS
    # ========================================================

    audio, _ = read_wav("transmission.wav")

    metrics = calculate_signal_metrics(audio)
    print_signal_metrics(metrics)

    # ========================================================
    # 3. RECEIVER
    # ========================================================

    recovered_message = receive("transmission.wav")

    # ========================================================
    # 4. FINAL RESULT
    # ========================================================

    print("\n====================================")
    print("             FINAL RESULT")
    print("====================================")

    if recovered_message == message:
        print("TRANSMISSION: SUCCESS")
        print("Original and recovered messages match.")
    else:
        print("TRANSMISSION: FAILED")
        print("Original and recovered messages differ.")


if __name__ == "__main__":
    main()
