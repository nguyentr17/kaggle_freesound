#!/usr/bin/python3.6

import glob, os, sys
from typing import *

import numpy as np, pandas as pd, scipy as sp
import keras

from sklearn.model_selection import train_test_split, KFold

from data_v2 import find_files, build_caches, fit_labels
from data_v2 import get_label_binarizer, SoundDatagen


Tensor = Any

TOPK = 3

PREDICT_ONLY    = False
ENABLE_KFOLD    = False
TEST_SIZE       = 0.2
KFOLDS          = 10

NUM_CLASSES     = 41
SAMPLE_RATE     = 44100

# Network hyperparameters
NUM_EPOCHS      = 50


def map3_metric(predict: np.array, ground_truth: np.array) -> float:
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
    code_version = os.path.splitext(os.path.basename(__file__))[0]
    return "../models/%s__%s.hdf5" % (code_version, name)

def get_best_model_path(name: str) -> str:
    """ Returns the path of the best model. """
    return get_model_path("%s_best" % name)


class Map3Metric(keras.callbacks.Callback):
    """ Keras callback that calculates MAP3 metric. """
    def __init__(self, datagen: SoundDatagen, y: np.array, name: str) -> None:
        self.datagen    = datagen
        self.y          = y
        self.name       = name

        self.best_map3  = 0.0
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Any = {}) -> None:
        predict = self.model.predict_generator(self.datagen)
        map3 = map3_metric(predict, self.y)
        print("epoch %d MAP@3 %.4f" % (epoch, map3))

        if map3 > self.best_map3:
            self.best_map3 = map3
            self.best_epoch = epoch

            self.model.save(get_model_path("%s_val_%.4f" % (self.name, map3)))
            self.model.save(get_best_model_path(self.name))

def train_model(train_datagen: SoundDatagen, val_datagen: SoundDatagen,
                labels: np.array, name: str) -> None:
    """ Creates model, trains it and saves it to the file. """
    shape = train_datagen.shape()
    x = inp = keras.layers.Input(shape=shape[1:])

    x = keras.layers.Convolution2D(32, (4,10), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Convolution2D(32, (4,10), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Convolution2D(32, (4,10), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Convolution2D(32, (4,10), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Convolution2D(32, (4,10), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Convolution2D(32, (4,10), padding="same")(x)
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

    map3 = Map3Metric(val_datagen, labels, name)
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    model.fit_generator(train_datagen, epochs=NUM_EPOCHS,
                        verbose=1, shuffle=False,
                        use_multiprocessing=False,
                        # use_multiprocessing=True, workers=12,
                        validation_data=val_datagen, callbacks=[map3])

    print("best MAP@3 value: %.04f at epoch %d" % (map3.best_map3, map3.best_epoch))

def merge_predictions(pred: np.array, mode: str, axis: int) -> np.array:
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

def predict(datagen: SoundDatagen, model_name: str) -> np.array:
    """ Predicts results on test, using the best model. """
    clips_per_sample = datagen.get_clips_per_sample()

    print("predicting results")
    model = keras.models.load_model(get_best_model_path(model_name))

    y_test = model.predict_generator(datagen)
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

def encode_predictions(y_test: np.array) -> np.array:
    """ Takes array NUM_SAMPLES * NUM_CLASSES, returns array NUM_SAMPLES
    of strings."""
    print("y_test.shape after merge", y_test.shape)
    label_binarizer = get_label_binarizer()

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
    train_files = np.array(find_files("../data/audio_train/"))
    assert(len(train_files) == 9473)

    train_labels = train_df["label"]
    test_files = np.array(find_files("../data/audio_test/"))
    test_idx = [os.path.basename(f) for f in test_files]

    build_caches(np.concatenate([train_files, test_files]))
    fit_labels(train_labels)

    if not ENABLE_KFOLD:
        x_train, x_val, y_train, y_val = train_test_split(
            train_files, train_labels, test_size=TEST_SIZE, shuffle=False)

        if not PREDICT_ONLY:
            train_model(SoundDatagen(x_train, y_train),
                        SoundDatagen(x_val, y_val), y_val, "nofolds")

        pred = predict(SoundDatagen(test_files, None), "nofolds")
        pred = encode_predictions(pred)
    else:
        kf = KFold(n_splits=KFOLDS, shuffle=False)
        pred = np.zeros((len(test_idx), KFOLDS, NUM_CLASSES))

        for k, (train_idx, val_idx) in enumerate(kf.split(train_files)):
            print("fold %d ==============================================" % k)
            name = "fold_%d" % k

            x_train, y_train = train_files[train_idx], train_labels[train_idx]
            x_val, y_val = train_files[val_idx], train_labels[val_idx]

            if not PREDICT_ONLY:
                train_model(SoundDatagen(x_train, y_train),
                            SoundDatagen(x_val, y_val), y_val, name)

            pred[:, k, :] = predict(SoundDatagen(test_files, None), name)

        print("before final merge: pred.shape", pred.shape)
        pred = merge_predictions(pred, "geom_mean", axis=1)
        print("predictions after final merge", pred.shape)
        pred = encode_predictions(pred)
        print("predictions after encoding", pred.shape)

    sub = pd.DataFrame({"fname": test_idx, "label": pred})
    version = os.path.splitext(os.path.basename(__file__))[0]
    sub.to_csv("../submissions/%s.csv" % version, index=False, header=True)
    print("submission has been generated")
