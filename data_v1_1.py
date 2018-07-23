#!/usr/bin/python3.6

import glob, multiprocessing, os, sys
from typing import *

import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm


DATA_VERSION    = 0x11

NpArray = Any

SAMPLE_RATE = 44100

# Data parameters
NUM_MFCC_ROWS       = 120
SILENCE_THRESHOLD   = 1e-1
MIN_DURATION        = 0.5
MAX_ALLOWED_SILENCE = 0.2
CLIP_DURATION       = 2

ENABLE_DEBUGGING    = False


def dprint(*args: Any) -> None:
    if ENABLE_DEBUGGING:
        print(*args)

def skip_leading_silence(waveform: NpArray) -> NpArray:
    """ Returns trimmed array. """
    dprint("skip_leading_silence")

    max_allowed_silence = int(MAX_ALLOWED_SILENCE * SAMPLE_RATE)

    for pos, level in enumerate(waveform):
        if abs(level) > SILENCE_THRESHOLD:
            base = max(pos, 0)
            dprint("skipping first %d samples" % base)
            return waveform[base:]

    # Signal is not detected, return an empty array.
    dprint("skip_leading_silence: end of data or complete silence")
    return np.array([])

def pad(a: NpArray) -> NpArray:
    max_clip_duration = int(CLIP_DURATION * SAMPLE_RATE)
    pad = max_clip_duration - a.size

    if pad > 0:
        a = np.pad(a, (0, pad), mode="constant", constant_values=(0, 0))

    return a

def extract_next_clip(waveform: NpArray) -> Tuple[NpArray, NpArray]:
    """ Scans the sound data until the maximum length is reached or silence
    is detected. Returns two arrays: signal and the rest of waveform. """
    dprint("extract_next_clip")

    max_allowed_silence = int(MAX_ALLOWED_SILENCE * SAMPLE_RATE)
    min_clip_duration   = int(MIN_DURATION * SAMPLE_RATE)
    max_clip_duration   = int(CLIP_DURATION * SAMPLE_RATE)
    dprint("max_clip_duration", max_clip_duration)
    dprint("waveform length", waveform.size)

    is_now_silence = False
    silence_duration = 0

    for pos, level in enumerate(waveform):
        silence = abs(level) < SILENCE_THRESHOLD

        if silence and is_now_silence:
            silence_duration += 1

            if silence_duration >= max_allowed_silence:
                dprint("returning first %d samples because of silence" % pos)
                return pad(waveform[:max_clip_duration]), waveform[pos:]
        elif silence and not is_now_silence:
            is_now_silence = True
            silence_duration = 1
        else:
            is_now_silence = False
            if pos >= max_clip_duration:
                dprint("returning first %d samples because of duration" % pos)
                return waveform[pos - max_clip_duration : pos], waveform[pos:]

    dprint("extract_next_clip: end of data")
    dprint("waveform.size", waveform.size, "min_clip_duration", min_clip_duration)

    if waveform.size < min_clip_duration:
        dprint("returning empty arrays")
        return np.array([]), np.array([])
    else:
        return pad(waveform[:max_clip_duration]), np.array([])

def break_waveform_into_clips(waveform: NpArray) -> List[NpArray]:
    """ Breaks waveform into clips with maximum length limit while skipping
    silence. """
    max_clip_duration = int(CLIP_DURATION * SAMPLE_RATE)
    clips = []

    waveform /= np.max(waveform)

    while waveform.size:
        waveform = skip_leading_silence(waveform)
        if not waveform.size:
            break

        signal, waveform = extract_next_clip(waveform)
        dprint("extract_next_clip returned %d samples" % signal.size)

        if signal.size:
            assert(signal.size == max_clip_duration)
            clips.append(signal)

    return clips

def visualize_random_file(test_file: str) -> None:
    """ Debug visualization. """
    waveform, sr = librosa.core.load(test_file, sr=SAMPLE_RATE)
    assert(sr == SAMPLE_RATE)

    print("analyzing file:", test_file)
    print("waveform duration:", waveform.size)

    clips = break_waveform_into_clips(waveform)
    cols = len(clips) + 1

    plt.subplot(cols + 1, 1, 1)
    plt.plot(waveform)

    for i, clip in enumerate(clips):
        plt.subplot(cols + 1, 1, 2 + i)
        plt.plot(clip)

    plt.xlabel(test_file)
    plt.show()

    sys.exit()

def read_file(path: str) -> List[NpArray]:
    """ Reads the wav file and returns a list of np.arrays. """
    waveform, sr = librosa.core.load(path, sr=SAMPLE_RATE)
    assert(sr == SAMPLE_RATE)
    data = []

    if waveform.size == 0:
        return []    # handle empty files

    for clip in break_waveform_into_clips(waveform):
        mfcc = librosa.feature.mfcc(clip, sr, n_mfcc=NUM_MFCC_ROWS)
        data.append(mfcc)

    return data

def load_dataset(files: List[str]) -> List[List[NpArray]]:
    """ Loads a particular dataset. """
    print("loading files, first 5 are:", files[:5])

    # test_file = np.random.choice(files)
    # test_file = "../data/audio_train/8f6071d2.wav"
    # visualize_random_file(test_file)

    pool = multiprocessing.Pool(processes=12)
    data = [x for x in tqdm(pool.imap(read_file, files), total=len(files))]
    return data
