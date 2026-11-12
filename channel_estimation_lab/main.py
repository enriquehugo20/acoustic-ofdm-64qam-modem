import argparse

from transmitter.audio_generator import generate_audio
from receiver.audio_recorder import record_audio
from receiver.channel_estimator import run_channel_estimation, run_self_tests
from receiver.channel_analyzer import analyze_channel


def main():
    parser = argparse.ArgumentParser(
        description='Acoustic OFDM channel estimation laboratory'
    )
    parser.add_argument(
        'operation', choices=['transmit', 'record', 'estimate', 'all', 'test']
    )
    operation = parser.parse_args().operation

    if operation == 'transmit':
        generate_audio()
    elif operation == 'record':
        record_audio()
    elif operation == 'estimate':
        response, taps = run_channel_estimation()
        analyze_channel(response, taps)
    elif operation == 'all':
        generate_audio()
        record_audio()
        response, taps = run_channel_estimation()
        analyze_channel(response, taps)
    elif operation == 'test':
        run_self_tests()


if __name__ == '__main__':
    main()
