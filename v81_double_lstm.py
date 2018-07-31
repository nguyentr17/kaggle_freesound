#!/usr/bin/python3.6

import glob, os, sys
from typing import *

import numpy as np, pandas as pd, scipy as sp
import keras

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold

from data_v1 import load_everything, load_data, map3_metric


CODE_VERSION    = os.path.splitext(os.path.basename(__file__))[0][1:]

NpArray         = Any
TOPK            = 3

PREDICT_ONLY    = False
ENABLE_KFOLD    = True
TEST_SIZE       = 0.2
KFOLDS          = 10

NUM_CLASSES     = 41
SAMPLE_RATE     = 44100

# Network hyperparameters
BATCH_SIZE      = 32
NUM_EPOCHS      = 70


def get_model_path(name: str) -> str:
    """ Builds the path of the model file. """
    return "../models/%s__%s.hdf5" % (CODE_VERSION, name)

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

        self.last_best_map3 = 0.0
        self.last_best_epoch = 0
        self.max_epochs = 15
        self.min_threshold = 0.001

    def on_epoch_end(self, epoch: int, logs: Any = {}) -> None:
        predict = self.model.predict(self.x_val)
        map3 = map3_metric(predict, self.y_val)
        print("epoch %d MAP@3 %.4f" % (epoch + 1, map3))

        if map3 > self.best_map3:
            self.best_map3 = map3
            self.best_epoch = epoch

            self.model.save(get_model_path("%s_val_%.4f" % (self.name, map3)))
            self.model.save(get_best_model_path(self.name))

        # Optionally do early stopping, basing on MAP@3 metric
        if self.best_map3 > self.last_best_map3 + self.min_threshold:
            self.last_best_map3 = map3
            self.last_best_epoch = epoch
        elif epoch >= self.last_best_epoch + self.max_epochs:
            self.model.stop_training = True
            print("stopping training because MAP@3 growth has stopped")

def train_model(params: Dict[str, Any], name: str ="nofolds") -> float:
    """ Creates model, trains it and saves it to the file. """
    shape = x_train.shape
    x = inp = keras.layers.Input(shape=shape[1:])

    x = keras.layers.Permute((2, 1))(x)

    x = keras.layers.Convolution1D(64, 1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.LSTM(128, return_sequences=True)(x)
    x = keras.layers.LSTM(128)(x)

    reg = None # keras.regularizers.l2(10 ** 4.5)
    x = keras.layers.Dense(128,
                           kernel_regularizer=reg,
                           bias_regularizer=reg,
                           activity_regularizer=reg,
                           )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    map3 = Map3Metric(x_val, y_val, name)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                    factor=0.33, patience=7, verbose=1, min_lr=3e-6)

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
              verbose=1, validation_data=[x_val, y_val],
              callbacks=[map3, reduce_lr])

    print("best MAP@3 value: %.04f at epoch %d" % (map3.best_map3,
                                                   map3.best_epoch + 1))
    return map3.best_map3

def merge_predictions(pred: NpArray, mode: str, axis: int) -> NpArray:
    """ Merges predictions for all clips and returns a single prediction. """
    assert(pred.shape[-1] == NUM_CLASSES)

    if mode == "mean":
        return np.mean(pred, axis=axis)
    elif mode == "max":
        return np.max(pred, axis=axis)
    elif mode == "geom_mean":
        # this code is same as in scipy, but it prevents warning
        res = 1

        for i in range(pred.shape[axis]):
            res = res * np.take(pred, i, axis=axis)

        return res ** (1 / pred.shape[axis])
    else:
        assert(False)

def predict(x_test: NpArray, label_binarizer: Any, clips_per_sample: List[int],
            model_name: str) -> NpArray:
    """ Predicts results on test, using the best model. """
    print("predicting results")
    model = keras.models.load_model(get_best_model_path(model_name))

    y_test = model.predict(x_test, verbose=1)
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

if __name__ == "__main__":
    train_indices, test_idx, train_labels, class_weights = load_everything()

    if not ENABLE_KFOLD:
        train_idx, val_idx = train_test_split(train_indices,
                                              stratify=train_labels,
                                              random_state=0,
                                              test_size=TEST_SIZE)
        x_train, y_train, x_val, y_val, x_test, label_binarizer, \
            clips_per_sample = load_data(train_idx, val_idx)

        if not PREDICT_ONLY:
            params: Dict[str, Any] = {
            }
            train_model(params)

        pred = predict(x_test, label_binarizer, clips_per_sample, "nofolds")
    else:
        kf = StratifiedKFold(n_splits=KFOLDS, shuffle=False)
        pred = np.zeros((len(test_idx), KFOLDS, NUM_CLASSES))

        for k, (train_idx, val_idx) in enumerate(kf.split(train_indices, train_labels)):
            print("fold %d ==============================================" % k)

            x_train, y_train, x_val, y_val, x_test, label_binarizer, \
                clips_per_sample = load_data(train_idx, val_idx)
            name = "fold_%d" % k

            if not PREDICT_ONLY:
                params = {
                }
                train_model(params, name=name)

            pred[:, k, :] = predict(x_test, label_binarizer, clips_per_sample, name)

        print("before final merge: pred.shape", pred.shape)
        pred = merge_predictions(pred, "geom_mean", axis=1)
        print("predictions after final merge", pred.shape)

    np.savez("../predictions/%s.npz" % CODE_VERSION, predict=pred)
    print("matrix of predictions has been saved")
