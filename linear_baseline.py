#!/usr/bin/python3.6

import glob, os, pickle, sys
from typing import *

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import librosa
import keras
from tqdm import tqdm

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

NpArray = Any
Tensor = Any

TOPK = 3

VERSION         = 0x01

MAX_MFCC_DEPTH  = 300
MAX_TRAIN_FILES = None
MAX_TEST_FILES  = None
TEST_SIZE       = 0.2
NUM_CLASSES     = 41

BATCH_SIZE  = 32
NUM_EPOCHS  = 200
NUM_HIDDEN  = 100


def find_files(path: str, max_files: Optional[int] = None) -> List[str]:
    """ Reads whole dataset into np.array. """
    files = sorted(list(glob.glob(os.path.join(path, "*.wav"))))

    if max_files:
        files = files[:max_files]

    print("files: %d files found" % len(files))
    return files

def load_dataset(files: List[str]) -> NpArray:
    """ Loads a particular dataset. """
    print("loading files, first 5 are:", files[:5])

    NUM_MFCC_ROWS = 20
    data = np.zeros((len(files), NUM_MFCC_ROWS, MAX_MFCC_DEPTH))

    for i, sample in enumerate(tqdm(files)):
        wave, sr = librosa.core.load(sample, sr=44100)
        assert(sr == 44100)

        if wave.size > 0:
            mfcc = librosa.feature.mfcc(wave, sr, n_mfcc=NUM_MFCC_ROWS)

            depth = min(MAX_MFCC_DEPTH, mfcc.shape[1])
            data[i, :, :depth] = mfcc[:, :depth]

    print("shape of data", data.shape)
    return data

def load_data() -> Tuple[NpArray, NpArray, NpArray, List[str], Any]:
    """ Loads all data. """
    train_df = pd.read_csv("../data/train.csv", index_col="fname")

    train_cache = "../output/train_cache.pkl"
    test_cache = "../output/test_cache.pkl"

    print("reading train dataset")
    train_files = find_files("../data/audio_train/", MAX_TRAIN_FILES)

    if os.path.exists(train_cache):
        x = pickle.load(open(train_cache, "rb"))
    else:
        x = load_dataset(train_files)
        pickle.dump(x, open(train_cache, "wb"))

    print("reading test dataset")
    test_files = find_files("../data/audio_test/", MAX_TEST_FILES)
    test_index = [os.path.basename(f) for f in test_files]

    if os.path.exists(test_cache):
        x_test = pickle.load(open(test_cache, "rb"))
    else:
        x_test = load_dataset(test_files)
        pickle.dump(x_test, open(test_cache, "wb"))

    mean, std = np.mean(x), np.std(x)
    x = (x - mean) / std
    x_test = (x_test - mean) / std

    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(train_df["label"])
    labels_dict = {file: labels[i] for i, file in enumerate(train_df.index)}
    y = np.array([labels_dict[os.path.basename(file)] for file in train_files])
    print("y", y.shape)

    return x, y, x_test, test_index, label_binarizer

def map3_metric(predict: NpArray, ground_truth: NpArray) -> float:
    """ Implements Mean Average Precision @ Top 3. """
    results = []
    assert(predict.shape[0] == ground_truth.shape[0])

    for actual, pred in zip(ground_truth, predict):
        actual = np.argmax(actual)

        pred = np.argsort(pred)[-TOPK:]
        pred = np.flip(pred, axis=0)

        score = 0.0
        num_hits = 0

        for i, p in enumerate(pred):
            if p == actual and p not in pred[:i]:
                num_hits += 1
                score += num_hits / (i + 1)

        results.append(score)

    assert(len(results) == predict.shape[0])
    return np.mean(results)

def get_model_path(name: str) -> str:
    """ Builds the path of the model file. """
    return "../models/%02x__%s.hdf5" % (VERSION, name)

def get_best_model_path() -> str:
    """ Returns the path of the best model. """
    return get_model_path("best")

class Map3Metric(keras.callbacks.Callback):
    """ Keras callback that calculates MAP3 metric. """
    def __init__(self, x_val: NpArray, y_val: NpArray) -> None:
        self.x_val = x_val
        self.y_val = y_val
        self.best_map3 = 0.0

    def on_epoch_end(self, epoch: int, logs: Any = {}) -> None:
        predict = self.model.predict(x_val)
        map3 = map3_metric(predict, y_val)
        print("epoch %d MAP@3 %.4f" % (epoch, map3))

        if map3 > self.best_map3:
            self.best_map3 = map3
            self.model.save(get_model_path("val_%.4f" % map3))
            self.model.save(get_best_model_path())

def train_model(x_train: NpArray, x_val: NpArray, y_train: NpArray, y_val:
                NpArray) -> Any:
    """ Creates model and trains it. Returns trained model. """
    # reg = keras.regularizers.l2(1e-3)
    reg = None

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=x_train.shape[1:]))
    model.add(keras.layers.Dense(NUM_HIDDEN, kernel_regularizer=reg,
                                 activation='relu'))
    # model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(NUM_CLASSES, kernel_regularizer=reg,
                                 activation='softmax'))

    model.summary()

    metrics = [Map3Metric(x_val, y_val)]
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
              verbose=2, validation_data=[x_val, y_val], callbacks=metrics)

    print("best MAP@3 value: %.04f" % metrics[0].best_map3)
    return model

def predict(x_test: NpArray) -> NpArray:
    """ Predicts results on test, using the best model. """
    print("predicting results")
    model = keras.models.load_model(get_best_model_path())

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

    return joined_pred

if __name__ == "__main__":
    x, y, x_test, test_index, label_binarizer = load_data()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=TEST_SIZE)
    print("x_train", x_train.shape)
    print("x_val", x_val.shape)
    print("y_train", y_train.shape)
    print("y_val", y_val.shape)

    # train_model(x_train, x_val, y_train, y_val)
    pred = predict(x_test)

    sub = pd.DataFrame({"fname": test_index, "label": pred})
    sub.to_csv("../submissions/%02x.csv" % VERSION, index=False, header=True)
    print("submission has been generated")
