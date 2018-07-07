#!/usr/bin/python3.6

import glob, os, pickle, sys
from typing import *

import numpy as np, pandas as pd, scipy as sp
import keras

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from data import load_dataset


NpArray = Any
Tensor = Any

TOPK = 3

CODE_VERSION    = 0x07
DATA_VERSION    = 0x01

PREDICT_ONLY    = False
ENABLE_KFOLD    = False
TEST_SIZE       = 0.2
KFOLDS          = 10

NUM_CLASSES     = 41
SAMPLE_RATE     = 44100

# Network hyperparameters
BATCH_SIZE      = 32
NUM_EPOCHS      = 50
NUM_HIDDEN1     = 200
NUM_HIDDEN2     = 200
DROPOUT_COEFF   = 0.1


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
    version = os.path.splitext(os.path.basename(__file__))[0]
    return "../models/%s__%s.hdf5" % (version, name)

def get_best_model_path(name: str) -> str:
    """ Returns the path of the best model. """
    return get_model_path("%s_best" % name)

class Map3Metric(keras.callbacks.Callback):
    """ Keras callback that calculates MAP3 metric. """
    def __init__(self, x_val: NpArray, y_val: NpArray, name: str) -> None:
        self.x_val = x_val
        self.y_val = y_val
        self.name = name

        self.best_map3 = 0.0
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Any = {}) -> None:
        predict = self.model.predict(self.x_val)
        map3 = map3_metric(predict, self.y_val)
        print("epoch %d MAP@3 %.4f" % (epoch, map3))

        if map3 > self.best_map3:
            self.best_map3 = map3
            self.best_epoch = epoch

            self.model.save(get_model_path("%s_val_%.4f" % (self.name, map3)))
            self.model.save(get_best_model_path(self.name))

def train_model(x_train: NpArray, x_val: NpArray, y_train: NpArray, y_val:
                NpArray, model_name: str) -> Any:
    """ Creates model and trains it. Returns trained model. """
    x = inp = keras.layers.Input(shape=x_train.shape[1:])
    x = keras.layers.Reshape((x_train.shape[1], x_train.shape[2], 1))(x)

    x = keras.layers.Convolution2D(32, (4,20), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Convolution2D(32, (4,20), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Convolution2D(32, (4,20), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Convolution2D(32, (4,20), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    model.summary()

    map3 = Map3Metric(x_val, y_val, model_name)
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
              verbose=0, validation_data=[x_val, y_val], callbacks=[map3])

    print("best MAP@3 value: %.04f at epoch %d" % (map3.best_map3, map3.best_epoch))
    return model

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
            model_name: str) -> NpArray:
    """ Predicts results on test, using the best model. """
    print("predicting results")
    model = keras.models.load_model(get_best_model_path(model_name))

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

    if not ENABLE_KFOLD:
        train_idx, val_idx = train_test_split(train_indices, test_size=TEST_SIZE)
        x_train, y_train, x_val, y_val, x_test, label_binarizer, \
            clips_per_sample = load_data(train_idx, val_idx)

        if not PREDICT_ONLY:
            train_model(x_train, x_val, y_train, y_val, "nofolds")

        pred = predict(x_test, label_binarizer, clips_per_sample, "nofolds")
        pred = encode_predictions(pred)
    else:
        kf = KFold(n_splits=KFOLDS, shuffle=False)
        pred = np.zeros((len(test_idx), KFOLDS, NUM_CLASSES))

        for k, (train_idx, val_idx) in enumerate(kf.split(train_indices)):
            print("fold %d ==============================================" % k)

            x_train, y_train, x_val, y_val, x_test, label_binarizer, \
                clips_per_sample = load_data(train_idx, val_idx)
            name = "fold_%d" % k

            if not PREDICT_ONLY:
                train_model(x_train, x_val, y_train, y_val, name)

            pred[:, k, :] = predict(x_test, label_binarizer, clips_per_sample, name)

        print("before final merge: pred.shape", pred.shape)
        pred = merge_predictions(pred, "geom_mean", axis=1)
        print("predictions after final merge", pred.shape)
        pred = encode_predictions(pred)
        print("predictions after encoding", pred.shape)

    sub = pd.DataFrame({"fname": test_idx, "label": pred})
    sub.to_csv("../submissions/%02x.csv" % CODE_VERSION, index=False, header=True)
    print("submission has been generated")
