#!/usr/bin/python3.6

import glob, multiprocessing, os, sys
from typing import *

import numpy as np
import librosa
import keras
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


def get_random_eraser(p: float = 0.5, s_l: float = 0.02, s_h: float = 0.4,
                      r_1: float = 0.3, r_2: float = 1/0.3, v_l: float = 0,
                      v_h: float = 1.0) -> Any:
    """ An augmentation that erases part of the image. """
    def eraser(input_img: NpArray) -> NpArray:
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

class MixupGenerator(keras.utils.Sequence):
    """ Implements mixup of audio clips. """
    def __init__(self, X_train: NpArray, y_train: NpArray, batch_size: int = 32,
                 alpha: float = 0.2, shuffle: bool = True, datagen: Any = None) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen
        self.indexes = self.__get_exploration_order()

    def __get_exploration_order(self) -> NpArray:
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __len__(self) -> int:
        return int(len(self.indexes) // (self.batch_size * 2))

    def __getitem__(self, i: int) -> np.array:
        batch_ids = self.indexes[i * self.batch_size * 2 :
                                 (i + 1) * self.batch_size * 2]
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
