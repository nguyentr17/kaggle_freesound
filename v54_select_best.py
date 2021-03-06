#!/usr/bin/python3.6

import glob, os, pickle, re, sys
from typing import *

import numpy as np, pandas as pd, scipy as sp
import keras

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight

from data import load_dataset


NpArray = Any

TOPK = 3

CODE_VERSION    = os.path.splitext(os.path.basename(__file__))[0][1:]
DATA_VERSION    = 0x01

KFOLDS          = 20
NUM_CLASSES     = 41


def find_files(path: str) -> List[str]:
    """ Reads whole dataset into np.array. """
    files = sorted(list(glob.glob(os.path.join(path, "*.wav"))))

    print("files: %d files found" % len(files))
    return files

def build_train_dataset(x: List[List[NpArray]], y: NpArray, indices:
                        List[int]) -> \
                        Tuple[NpArray, NpArray]:
    """ Builds dataset with multiple items per sample. Returns (x, y). """
    res_x, res_y = [], []

    for idx in indices:
        for sample in x[idx]:
            res_x.append(sample)
            res_y.append(y[idx])

    return np.array(res_x), np.array(res_y)

def build_test_dataset(x: List[List[NpArray]]) -> Tuple[NpArray, List[int]]:
    """ Builds test dataset with multiple clips per sample, which will
    be joined later somehow (maximum, geom. mean, etc).
    Returns: (x_test, clips_per_sample). """
    x_test, clips_per_sample = [], []

    for sample in x:
        clips_per_sample.append(len(sample))

        for clip in sample:
            x_test.append(clip)

    return np.array(x_test), clips_per_sample

def load_data(train_idx: NpArray, val_idx: NpArray) -> \
              Tuple[NpArray, NpArray, NpArray, NpArray, NpArray, Any, List[int]]:
    """ Loads all data. """
    train_df = pd.read_csv("../data/train.csv", index_col="fname")

    train_cache = "../output/train_cache_v%02d.pkl" % DATA_VERSION
    test_cache = "../output/test_cache_v%02d.pkl" % DATA_VERSION

    print("reading train dataset")
    if os.path.exists(train_cache):
        x = pickle.load(open(train_cache, "rb"))
    else:
        train_files = find_files("../data/audio_train/")
        print("len(train_files)", len(train_files))

        x = load_dataset(train_files)
        pickle.dump(x, open(train_cache, "wb"))

    # get whatever data metrics we need
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(train_df["label"])

    x_train, y_train = build_train_dataset(x, train_df["label"], train_idx)
    x_val, y_val = build_train_dataset(x, train_df["label"], val_idx)

    print("reading test dataset")
    if os.path.exists(test_cache):
        x_test = pickle.load(open(test_cache, "rb"))
    else:
        test_files = find_files("../data/audio_test/")
        print("len(test_files)", len(test_files))

        x_test = load_dataset(test_files)
        pickle.dump(x_test, open(test_cache, "wb"))

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

def get_best_model_path(fold: int) -> str:
    """ Looks for all files *__file_NUMBER_ and chooses one with best acc. """
    def parse_accuracy(filename: str) -> float:
        m = re.search(r"__fold_\d+_val_([01]\.\d+)", filename)
        assert(m)
        return float(m.group(1))

    models = list(glob.glob("../models/*__fold_%d_val_*.hdf5" % fold))
    accuracy = list(map(parse_accuracy, models))
    best = accuracy.index(max(accuracy))

    print("fold=%d best_model=%s" % (fold, models[best]))
    return models[best]

def merge_predictions(pred: NpArray, mode: str, axis: int) -> NpArray:
    """ Merges predictions for all clips and returns a single prediction. """
    assert(pred.shape[-1] == NUM_CLASSES)

    if mode == "mean":
        return np.mean(pred, axis=axis)
    elif mode == "max":
        return np.max(pred, axis=axis)
    elif mode == "geom_mean":
        return sp.stats.gmean(pred, axis=axis)
    else:
        assert(False)

def predict(x_test: NpArray, label_binarizer: Any, clips_per_sample: List[int],
            fold: int) -> NpArray:
    """ Predicts results on test, using the best model. """
    print("predicting results")
    model = keras.models.load_model(get_best_model_path(fold))

    y_test = model.predict(x_test)
    print("y_test.shape after predict", y_test.shape)

    pos = 0
    y_merged = []

    for count in clips_per_sample:
        if count != 0:
            y_merged.append(merge_predictions(y_test[pos : pos+count], "max", 0))
        else:
            y_merged.append(np.zeros_like(y_merged[0]))

        pos += count

    y_test = np.array(y_merged)
    print("y_test.shape after merge", y_test.shape)
    return y_test

def encode_predictions(y_test: NpArray) -> NpArray:
    """ Takes array NUM_SAMPLES x NUM_CLASSES, returns array NUM_SAMPLES
    of strings."""
    print("y_test.shape after merge", y_test.shape)

    # extract top K values
    y_test = np.argsort(y_test, axis=1)[:, -TOPK:]
    y_test = np.flip(y_test, axis=1)
    print("y_test", y_test.shape)

    n_test = y_test.shape[0]
    pred = np.zeros((n_test, TOPK), dtype=object)

    for col in range(TOPK):
        answers = keras.utils.to_categorical(y_test[:, col])
        pred[:, col] = label_binarizer.inverse_transform(answers)

    joined_pred = np.zeros(n_test, dtype=object)
    for row in range(n_test):
        joined_pred[row] = " ".join(pred[row, :])

    return joined_pred

if __name__ == "__main__":
    train_df = pd.read_csv("../data/train.csv", index_col="fname")
    train_indices = range(train_df.shape[0])

    test_files = find_files("../data/audio_test/")
    test_idx = [os.path.basename(f) for f in test_files]

    train_labels = stratify=train_df["label"]
    class_weight = class_weight.compute_class_weight('balanced',
                       np.unique(train_labels), train_labels)
    class_weight = {i: w for i, w in enumerate(class_weight)}
    print(class_weight)

    pred = np.zeros((len(test_idx), KFOLDS, NUM_CLASSES))
    kf = StratifiedKFold(n_splits=KFOLDS, shuffle=False)

    for k, (train_idx, val_idx) in enumerate(kf.split(train_indices, train_labels)):
        print("fold %d ==============================================" % k)

        # we only need x_test here
        x_train, y_train, x_val, y_val, x_test, label_binarizer, \
            clips_per_sample = load_data(train_idx, val_idx)

        pred[:, k, :] = predict(x_test, label_binarizer, clips_per_sample, k)

    print("before final merge: pred.shape", pred.shape)
    pred = merge_predictions(pred, "geom_mean", axis=1)
    print("predictions after final merge", pred.shape)
    pred = encode_predictions(pred)
    print("predictions after encoding", pred.shape)

    sub = pd.DataFrame({"fname": test_idx, "label": pred})
    sub.to_csv("../submissions/%s.csv" % CODE_VERSION, index=False, header=True)
    print("submission has been generated")
