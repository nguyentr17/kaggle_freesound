#!/usr/bin/python3.6

import glob, os
from typing import *

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import librosa
import keras

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

NpArray = Any

TOPK = 3

MAX_MFCC_DEPTH  = 300
MAX_TRAIN_FILES = 100
MAX_TEST_FILES  = 100
TEST_SIZE       = 0.2
NUM_CLASSES     = 41

BATCH_SIZE = 32
NUM_EPOCHS = 1

def find_files(path: str, max_files: int = 0) -> List[str]:
    """ Reads whole dataset into np.array. """
    files = sorted(list(glob.glob(os.path.join(path, "*.wav"))))

    if max_files:
        files = files[:max_files]

    print("files: %d files found" % len(files))
    return files

def load_dataset(files: List[str]) -> NpArray:
    print("loading files, first 5 are:", files[:5])

    NUM_MFCC_ROWS = 20
    data = np.zeros((len(files), NUM_MFCC_ROWS, MAX_MFCC_DEPTH))

    for i, sample in enumerate(files):
        # print("reading file '%s'" % sample)

        wave, sr = librosa.core.load(sample, sr=None)
        assert(sr == 44100)
        # print("sound data shape:", wave.shape)

        mfcc = librosa.feature.mfcc(wave, sr, n_mfcc=NUM_MFCC_ROWS)
        # print("mfcc shape:", mfcc.shape)

        depth = min(MAX_MFCC_DEPTH, mfcc.shape[1])
        data[i, :, :depth] = mfcc[:, :depth]

    print("shape of data", data.shape)
    return data

def map3_metric(predict: NpArray, ground_truth: NpArray) -> float:
    """ Implements Mean Average Precision @ Top 3. """
    results = []
    assert(predict.shape[0] == ground_truth.shape[0])

    for actual, predicted in zip(ground_truth, predict):
        actual, predicted = list(actual), list(predicted)

        # if len(predicted) > TOPK:
        predicted = predicted[:TOPK]

        score = 0.0
        num_hits = 0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1
                score += num_hits / (i + 1)

        if not actual:
            results.append(0.0)
        else:
            results.append(score / min(len(actual), TOPK))

    assert(len(results) == predict.shape[0])
    return np.mean(results)

def train_model(x_train: NpArray, x_val: NpArray, y_train: NpArray, y_val:
                NpArray) -> Any:
    """ Creates model and trains it. Returns trained model. """
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=x_train.shape[1:]))
    model.add(keras.layers.Dense(NUM_CLASSES))

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
                  # metrics=[map3_metric, "accuracy"])

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
              verbose=2, validation_data=[x_val, y_val])

    return model

if __name__ == "__main__":
    train_df = pd.read_csv("../data/train.csv", index_col="fname")

    print("reading train dataset")
    train_files = find_files("../data/audio_train/", MAX_TRAIN_FILES)
    x = load_dataset(train_files)

    print("reading test dataset")
    test_files = find_files("../data/audio_test/", MAX_TEST_FILES)
    test_index = [os.path.basename(f) for f in test_files]
    x_test = load_dataset(test_files)

    mean, std = np.mean(x), np.std(x)
    x = (x - mean) / std
    x_test = (x_test - mean) / std

    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(train_df["label"])
    labels_dict = {file: labels[i] for i, file in enumerate(train_df.index)}
    y = np.array([labels_dict[os.path.basename(file)] for file in train_files])

    print("y", y.shape)
    print(y[:5])

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=TEST_SIZE)
    print("x_train", x_train.shape)
    print("x_val", x_val.shape)
    print("y_train", y_train.shape)
    print("y_val", y_val.shape)

    model = train_model(x_train, x_val, y_train, y_val)

    print("predicting results")
    y_test = model.predict(x_test)
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

    sub = pd.DataFrame({"fname": test_index, "label": joined_pred})
    sub.to_csv("../submissions/submission.csv", index=False, header=True)
    print("submission has been generated")
