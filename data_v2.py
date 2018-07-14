#!/usr/bin/python3.6

import bisect, glob, itertools, multiprocessing, pickle, os, sys
from typing import *
from functools import partial

import numpy as np, pandas as pd, scipy as sp
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import LabelBinarizer
import keras


SAMPLE_RATE = 44100

# Data parameters
SILENCE_THRESHOLD   = 1e-1
MIN_DURATION        = 0.5
MAX_ALLOWED_SILENCE = 0.2
CLIP_DURATION       = 2

MIN_AMP             = 1e-10
BATCH_SIZE          = 32
NUM_CLASSES         = 41

ENABLE_DEBUGGING    = False

DATA_VERSION = os.path.splitext(os.path.basename(__file__))[0]


def dprint(*args: Any) -> None:
    if ENABLE_DEBUGGING:
        print(*args)

def skip_leading_silence(waveform: np.array) -> np.array:
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

def pad(a: np.array) -> np.array:
    max_clip_duration = int(CLIP_DURATION * SAMPLE_RATE)
    pad = max_clip_duration - a.size

    if pad > 0:
        a = np.pad(a, (0, pad), mode="constant", constant_values=(0, 0))

    return a

def extract_next_clip(waveform: np.array) -> Tuple[np.array, np.array]:
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

def break_waveform_into_clips(waveform: np.array) -> List[np.array]:
    """ Breaks waveform into clips with maximum length limit while skipping
    silence. Returns a list of waveforms, zero or more of them. """
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

def read_file(path: str) -> List[np.array]:
    """ Reads the wav file and returns a list of np.arrays. """
    waveform, sr = librosa.core.load(path, sr=SAMPLE_RATE)
    assert(sr == SAMPLE_RATE)
    data = []

    if waveform.size == 0:
        return []    # handle empty files

    for clip in break_waveform_into_clips(waveform):
        fft = librosa.core.stft(clip)

        abs = np.expand_dims(np.absolute(fft), axis=2)
        abs = np.log(np.maximum(abs, MIN_AMP))
        phase = np.expand_dims(np.angle(fft), axis=2)
        res = np.concatenate([abs, phase], axis=2)
        res = np.array(res, dtype=np.float32)

        data.append(res)

    return data

def find_files(path: str) -> List[str]:
    """ Returns list of wav files in a given directory. """
    files = sorted(list(glob.glob(os.path.join(path, "*.wav"))))

    print("files: %d files found" % len(files))
    return files

def get_cache_path(path: str) -> str:
    return "../data/cache/%s.fft" % os.path.basename(path)

def cache_file(path: str) -> int:
    """ Puts file in cache. Returns 0 or 1 to count files we changed. """
    cache_path = get_cache_path(path)
    if os.path.exists(cache_path):
        return 0

    data = read_file(path)
    pickle.dump(data, open(cache_path, "wb"))
    return 1

def get_num_clips(path: str) -> int:
    """ Returns number of clips in file. """
    cache_path = get_cache_path(path)
    return len(pickle.load(open(cache_path, "rb")))

def read_cached_file(path: str) -> np.array:
    """ Reads file from the cache. Returns same data as read_file(),
    cast to a single np.array. """
    cache_path = get_cache_path(path)
    return pickle.load(open(cache_path, "rb"))


data_mean, data_std = 0, 0
data_clips_per_sample: Dict[str, int] = dict()

def get_mean(path: str) -> np.array:
    """ Calculates mean of sound data. """
    data = pickle.load(open(get_cache_path(path), "rb"))
    data = np.array(data)
    if len(data.shape) != 4:
        return np.array((0, 0))

    mean = np.mean(data, axis=tuple([0, 1, 2]))
    return mean

def get_std(path: str, mean: np.array) -> np.array:
    data = pickle.load(open(get_cache_path(path), "rb"))
    data = np.array(data)
    if len(data.shape) != 4:
        return np.array((1, 1))

    std = np.sqrt(np.mean((data - mean) ** 2, axis=tuple([0, 1, 2])))
    return std

def build_caches(files: np.array) -> None:
    """ Ensures that caches are build both for train and test datasets. """
    pool = multiprocessing.Pool(processes=12)

    print("updating file cache...")
    count = sum(tqdm(pool.imap(cache_file, files), total=len(files)))
    print("%d files updated" % count)

    moments_cache = "../output/moments_%s.pickle" % DATA_VERSION
    global data_mean, data_std

    if os.path.exists(moments_cache):
        data_mean, data_std = pickle.load(open(moments_cache, "rb"))
    else:
        print("calculating dataset mean")
        means = list(tqdm(pool.imap(get_mean, files), total=len(files)))
        data_mean = np.mean(np.array(means), axis=0)

        print("calculating dataset std")
        part_func = partial(get_std, mean=data_mean)
        stds = list(tqdm(pool.imap(part_func, files), total=len(files)))
        data_std = np.mean(np.array(stds), axis=0)

        pickle.dump((data_mean, data_std), open(moments_cache, "wb"))

    print("data_mean", data_mean, "data_std", data_std)

    count_cache = "../output/clips_per_sample_%s.pickle" % DATA_VERSION
    global data_clips_per_sample

    if os.path.exists(count_cache):
        data_clips_per_sample = pickle.load(open(count_cache, "rb"))
    else:
        print("calculating number of clips per sample")
        count = list(tqdm(pool.imap(get_num_clips, files), total=len(files)))
        data_clips_per_sample = {f: count[i] for i, f in enumerate(files)}
        pickle.dump(data_clips_per_sample, open(count_cache, "wb"))


label_binarizer = LabelBinarizer()

def fit_labels(labels: np.array) -> None:
    """ Fits label binarizer. """
    label_binarizer.fit(labels)

def get_label_binarizer() -> LabelBinarizer:
    return label_binarizer


class Fragment:
    def __init__(self, file: str, base: int, clip_count: int,
                 label: np.array) -> None:
        """ Note that label is np.array (1, NUM_CLASSES). """
        self.file       = file
        self.base       = base
        self.clip_count = clip_count
        self.label      = label

class SoundDatagen(keras.utils.Sequence):
    """ Data generator for sound data with FFT or similar transform. """
    def __init__(self, x: List[str], y: Optional[List[str]]) -> None:
        """ Constructor. """
        self.files = x
        self.labels = label_binarizer.transform(y)
        self.batches = self.generate_table()
        self.last_file_name = ""
        self.last_file_data = np.empty(0)

    def generate_table(self) -> List[List[Fragment]]:
        """ Generates table with clip list for every batch.
        Returns list of batches, every batch is a list of Fragments. """
        clips_per_sample = [data_clips_per_sample[s] for s in self.files]
        batches = []
        reminder: List[Fragment] = []
        num_remaining_clips = 0
        print("clips_per_sample", clips_per_sample[:100])

        for file, clip_count, label in zip(self.files, clips_per_sample,
                                           self.labels):
            print("file=%s clip_count=%d" % (file, clip_count))
            base = 0

            # generate as many full batches as we can
            while num_remaining_clips + clip_count - base >= BATCH_SIZE:
                print("\tnum_remaining_clips=%d clip_count=%d base=%d" %
                      (num_remaining_clips, clip_count, base))
                count = BATCH_SIZE - num_remaining_clips
                print("\tappending clips (first=%d count=%d)" % (base, count))
                reminder.append(Fragment(file, base, count, label))

                assert(count + num_remaining_clips == BATCH_SIZE)
                batches.append(reminder)

                s = 0
                print("counting clips")
                
                for frag in batches[-1]:
                    print("\t", frag.file, frag.base, frag.clip_count)
                    s += frag.clip_count

                print("total clips count", s)
                assert(s == BATCH_SIZE)

                # print(batches[-1])
                # assert(sum(map(lambda f: f.clip_count, batches[-1])) == BATCH_SIZE)

                base += count
                reminder, num_remaining_clips = [], 0

            # put any unused clips into reminder
            reminder.append(Fragment(file, 0, clip_count, label))
            num_remaining_clips += clip_count - base
            print("\treminder: adding (base=%d count=%d) total=%d" %
                  (base, clip_count, num_remaining_clips))

        for i, batch in enumerate(batches):
            print("batch %d" % i)
            assert(sum(map(lambda f: f.clip_count, batch)) == BATCH_SIZE)

        # generate last incomplete batch
        if len(reminder):
            print("appending last reminder, count=%d" % num_remaining_clips)
            batches.append(reminder)

        return batches

    def __getitem__(self, idx: int) -> np.array:
        """ Returns a batch. """
        print("-------------------------\ngetitem(%d)" % idx)
        batch = self.batches[idx]
        x, y = [], []

        for frag in batch:
            if frag.file == self.last_file_name:
                clip = self.last_file_data
            else:
                self.last_file_data = clip = read_cached_file(frag.file)
                self.last_file_name = frag.file

            x.append(clip[frag.base : frag.base + frag.clip_count])
            y.extend([frag.label] * frag.clip_count)

            # print("label", frag.label.shape)
            # print("y", y)

        x_arr, y_arr = np.concatenate(x), np.array(y)
        print("x", len(x), "y", len(y))
        print("x_arr", x_arr.shape, "y_arr", y_arr.shape)

        assert(x_arr.shape[0] == BATCH_SIZE or idx == len(self) - 1)
        assert(x_arr.shape[0] == y_arr.shape[0])
        return x_arr, y_arr

    def __len__(self) -> int:
        """ Returns number of batches. """
        return len(self.batches)

    def on_epoch_end(self) -> None:
        """ Might be useful. """
        pass

    def shape(self) -> Tuple[int, int, int, int]:
        """ Returns shape of the X tensor. """
        some_x, some_y = self.__getitem__(0)
        return some_x.shape

    def get_clips_per_sample(self) -> List[int]:
        """ Returns number of clips for every sample. """
        return self.clips_per_sample
