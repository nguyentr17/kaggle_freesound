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

'''
# def load_dataset(files: List[str]) -> List[List[np.array]]:
#     """ Loads a particular dataset. """
#     print("loading files, first 5 are:", files[:5])
#
#     # test_file = np.random.choice(files)
#     # test_file = "../data/audio_train/8f6071d2.wav"
#     # visualize_random_file(test_file)
#
#     pool = multiprocessing.Pool(processes=12)
#     data = [x for x in tqdm(pool.imap(read_file, files), total=len(files))]
#     return data

def build_train_dataset(x: List[List[np.array]], y: np.array, indices:
                        List[int]) -> Tuple[np.array, np.array]:
    """ Builds dataset with multiple items per sample. Returns (x, y). """
    res_x, res_y = [], []

    for idx in indices:
        for sample in x[idx]:
            res_x.append(sample)
            res_y.append(y[idx])

    return np.array(res_x), np.array(res_y)

def build_test_dataset(x: List[List[np.array]]) -> Tuple[np.array, List[int]]:
    """ Builds test dataset with multiple clips per sample, which will
    be joined later somehow (maximum, geom. mean, etc).
    Returns: (x_test, clips_per_sample). """
    x_test, clips_per_sample = [], []

    for sample in x:
        clips_per_sample.append(len(sample))

        for clip in sample:
            x_test.append(clip)

    return np.array(x_test), clips_per_sample

def load_data(train_idx: np.array, val_idx: np.array) -> \
              Tuple[np.array, np.array, np.array, np.array, np.array, Any, List[int]]:
    """ Loads all data. """
    train_df = pd.read_csv("../data/train.csv", index_col="fname")

    train_cache = "../output/%s_train.pkl" % DATA_VERSION
    test_cache = "../output/%s_test.pkl" % DATA_VERSION

    print("reading train dataset")
    # if os.path.exists(train_cache):
    #     x = pickle.load(open(train_cache, "rb"))
    # else:

    train_files = find_files("../data/audio_train/")
    print("len(train_files)", len(train_files))
    x = load_dataset(train_files)

    # pickle.dump(x, open(train_cache, "wb"))

    # get whatever data metrics we need
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(train_df["label"])

    x_train, y_train = build_train_dataset(x, train_df["label"], train_idx)
    x_val, y_val = build_train_dataset(x, train_df["label"], val_idx)

    print("reading test dataset")
    # if os.path.exists(test_cache):
    #     x_test = pickle.load(open(test_cache, "rb"))
    # else:

    test_files = find_files("../data/audio_test/")
    print("len(test_files)", len(test_files))
    x_test = load_dataset(test_files)

    # pickle.dump(x_test, open(test_cache, "wb"))

    x_test, clips_per_sample = build_test_dataset(x_test)

    x_joined = np.concatenate([x_train, x_val])
    mean, std = np.mean(x_joined), np.std(x_joined)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    y_train = label_binarizer.transform(y_train)
    y_val = label_binarizer.transform(y_val)

    print("x_train.shape", x_train.shape, "y_train.shape", y_train.shape)
    print("x_val.shape", x_val.shape, "y_val.shape", y_val.shape)

    return x_train, y_train, x_val, y_val, x_test, label_binarizer, clips_per_sample
'''

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
    """ Reads file from the cache. """
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


class SoundDatagen(keras.utils.Sequence):
    """ Data generator for sound data with FFT or similar transform. """
    def __init__(self, x: List[str], y: Optional[List[str]]) -> None:
        """ Constructor. """
        self.files, self.labels = x, np.array(y)
        self.clips_per_sample = [data_clips_per_sample[s] for s in x]
        self.clip_offsets = list(itertools.accumulate(self.clips_per_sample))
        self.clip_offsets.insert(0, 0)

        print("clips_per_sample", self.clips_per_sample[:100])
        print("clip_offsets", self.clip_offsets[:100])

        self.clips_count = self.clip_offsets[-1]
        self.batch_count = int(np.ceil(self.clips_count / BATCH_SIZE))

    def __getitem__(self, idx: int) -> np.array:
        """ Returns a batch. """
        print("getitem(%d)" % idx)

        start = bisect.bisect_left(self.clip_offsets, idx * BATCH_SIZE)
        end = bisect.bisect_right(self.clip_offsets, (idx + 1) * BATCH_SIZE) + 1

        assert(start < len(self.clip_offsets))
        assert(end <= len(self.clip_offsets))
        print("start", start, "end", end)

        # concatenate all lists of clips into a single list
        clips = sum((read_cached_file(self.files[i]) for i in
                     range(start, end)), [])

        x = np.array(clips)
        x = (x - data_mean) / data_std
        print("x", x.shape)

        if self.labels is not None:
            y = self.labels[self.clip_offsets[start] : self.clip_offsets[end]]
            y = label_binarizer.transform(y)
            print("y", y.shape)

            assert(x.shape[0] == y.shape[0])
            return x, y
        else:
            return x, None

    def __len__(self) -> int:
        """ Returns number of batches. """
        return self.batch_count

    def on_epoch_end(self) -> None:
        """ Might be useful. """
        pass

    def shape(self) -> Tuple[int, int, int, int]:
        """ Returns shape of the X tensor. """
        item0 = self.__getitem__(0)
        return item0[0].shape

    def get_clips_per_sample(self) -> List[int]:
        """ Returns number of clips for every sample. """
        return self.clips_per_sample
