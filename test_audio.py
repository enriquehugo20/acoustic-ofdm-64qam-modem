
from receiver.audio_input import read_wav
from receiver.receiver import receive
from receiver.signal_metrics import (
    calculate_signal_metrics,
    print_signal_metrics,
)

AUDIO_FILE = r"C:\Users\enriq\workspace\acoustic-5g-modem\5G.wav"

print("\n===== REAL AUDIO CAPTURE =====")

audio, sample_rate = read_wav(AUDIO_FILE)

metrics = calculate_signal_metrics(audio)
print_signal_metrics(metrics)

print("\n===== OFDM REAL AUDIO DECODING =====")

recovered_message = receive(AUDIO_FILE)

print("\n===== RECOVERED MESSAGE =====")
print(recovered_message)
