import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from config import SAMPLE_RATE, OUTPUT_DIR


NUMBER_OF_CAPTURES = 3
PRE_RECORD_SECONDS = 1.0
POST_RECORD_SECONDS = 0.8
PAUSE_BETWEEN_CAPTURES = 1.0


def _load_transmission():
    path = OUTPUT_DIR / 'training_transmission.wav'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Run: python .\\main.py transmit'
        )
    fs, raw = wavfile.read(path)
    if fs != SAMPLE_RATE:
        raise ValueError(f'Transmission must be {SAMPLE_RATE} Hz; got {fs} Hz.')
    raw = np.asarray(raw)
    if raw.ndim == 2:
        raw = np.mean(raw.astype(np.float64), axis=1)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        if np.issubdtype(raw.dtype, np.unsignedinteger):
            midpoint = (info.max + 1) / 2.0
            audio = (raw.astype(np.float64) - midpoint) / midpoint
        else:
            audio = raw.astype(np.float64) / max(abs(info.min), info.max)
    else:
        audio = raw.astype(np.float64)
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError('Training WAV is empty or contains NaN/Inf.')
    return audio.astype(np.float32)


def _capture_once(tx_audio, number):
    chunks = []
    statuses = []

    def callback(indata, frames, time_info, status):
        if status:
            statuses.append(str(status))
        chunks.append(indata.copy())

    print(f'\nCAPTURE {number}/{NUMBER_OF_CAPTURES}')
    print('-' * 48)
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=callback,
        ):
            print(f'Microphone ready; pre-roll {PRE_RECORD_SECONDS:.1f} s')
            time.sleep(PRE_RECORD_SECONDS)
            print('Playing complete training burst...')
            sd.play(tx_audio, samplerate=SAMPLE_RATE, blocking=True)
            time.sleep(POST_RECORD_SECONDS)
    except Exception as exc:
        raise RuntimeError(f'Audio capture {number} failed: {exc}') from exc

    if not chunks:
        raise RuntimeError(f'Capture {number} returned no microphone samples.')
    recording = np.concatenate(chunks, axis=0).reshape(-1)
    if not np.all(np.isfinite(recording)):
        raise RuntimeError(f'Capture {number} contains NaN/Inf; file not saved.')

    peak = float(np.max(np.abs(recording)))
    rms = float(np.sqrt(np.mean(recording ** 2)))
    if peak < 1e-7 or rms < 1e-8:
        raise RuntimeError(
            f'Capture {number} is silent (peak={peak:.3e}, RMS={rms:.3e}).'
        )

    path = OUTPUT_DIR / f'received_recording_{number:02d}.wav'
    wavfile.write(path, SAMPLE_RATE, recording.astype(np.float32))
    print(f'Samples: {recording.size} | Duration: {recording.size / SAMPLE_RATE:.3f} s')
    print(f'Peak: {peak:.6f} | RMS: {rms:.6f} | NaN: 0 | Inf: 0')
    print(f'Saved: {path}')
    if statuses:
        print('Audio device warnings:')
        for item in statuses:
            print(f'  {item}')
    return path


def record_audio():
    """Record three independent full-burst captures automatically."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tx_audio = _load_transmission()
    print('\nAUTOMATED THREE-CAPTURE RECORDING')
    print('=' * 48)
    print(f'Sample rate: {SAMPLE_RATE} Hz')
    print('Keep speaker, microphone, volume and device settings unchanged.')
    paths = []
    for number in range(1, NUMBER_OF_CAPTURES + 1):
        paths.append(_capture_once(tx_audio, number))
        if number < NUMBER_OF_CAPTURES:
            print(f'Pause {PAUSE_BETWEEN_CAPTURES:.1f} s...')
            time.sleep(PAUSE_BETWEEN_CAPTURES)
    print('\nAll three recordings completed.')
    return paths
