#!/usr/bin/python3.6

import os,sys
from typing import *

import numpy as np, pandas as pd, scipy as sp
import keras
from sklearn.preprocessing import LabelBinarizer
from data_v2 import find_files

NpArray = Any

NUM_CLASSES     = 41
TOPK            = 3


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
    if len(sys.argv) < 3 or not sys.argv[1].endswith(".csv"):
        print("usage: %s <dest.csv> <file1.npz> <file2.npz> ..." % sys.argv[0])
        sys.exit()

    train_df = pd.read_csv("../data/train.csv", index_col="fname")
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(train_df["label"])

    test_files = np.array(find_files("../data/audio_test/"))
    test_idx = [os.path.basename(f) for f in test_files]

    dest_filename = sys.argv[1]
    predict_count = len(sys.argv[2:])
    pred = np.zeros((len(test_idx), predict_count, NUM_CLASSES))

    for k, pred_name in enumerate(sys.argv[2:]):
        print("reading", pred_name)
        pred[:, k, :] = np.load(pred_name)["predict"]

    pred = merge_predictions(pred, "geom_mean", axis=1)
    print("predictions after final merge", pred.shape)

    pred = encode_predictions(pred)
    print("predictions after encoding", pred.shape)

    print("writing results")
    sub = pd.DataFrame({"fname": test_idx, "label": pred})
    sub.to_csv(dest_filename, index=False, header=True)
    print("submission has been generated")
