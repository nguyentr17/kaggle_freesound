#!/usr/bin/python3.6

import glob, os, pickle, sys
from typing import *

import numpy as np, pandas as pd
import keras

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from data import load_dataset


NpArray = Any
Tensor = Any

TOPK = 3

CODE_VERSION    = 0x03
DATA_VERSION    = 0x01
PREDICT_ONLY    = False

TEST_SIZE       = 0.2
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

def take_first(x: List[List[NpArray]]) -> NpArray:
    """ Creates a tensor from list of lists. """
    null = np.zeros_like(x[0][0])
    singles = [L[0] if L else null for L in x]
    res = np.array(singles)
    print("res.shape", res.shape)
    return res

def load_data() -> Tuple[NpArray, NpArray, NpArray, List[str], Any]:
    """ Loads all data. """
    train_df = pd.read_csv("../data/train.csv", index_col="fname")

    train_cache = "../output/train_cache_v%02d.pkl" % DATA_VERSION
    test_cache = "../output/test_cache_v%02d.pkl" % DATA_VERSION

    print("reading train dataset")
    train_files = find_files("../data/audio_train/")
    print("len(train_files)", len(train_files))

    if os.path.exists(train_cache):
        x = pickle.load(open(train_cache, "rb"))
    else:
        x = load_dataset(train_files)
        pickle.dump(x, open(train_cache, "wb"))

    x = take_first(x)
    print("x.shape", x.shape)

    print("reading test dataset")
    test_files = find_files("../data/audio_test/")
    test_index = [os.path.basename(f) for f in test_files]

    if os.path.exists(test_cache):
        x_test = pickle.load(open(test_cache, "rb"))
    else:
        x_test = load_dataset(test_files)
        pickle.dump(x_test, open(test_cache, "wb"))

    x_test = take_first(x_test)
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
    return "../models/%02x__%s.hdf5" % (CODE_VERSION, name)

def get_best_model_path() -> str:
    """ Returns the path of the best model. """
    return get_model_path("best")

class Map3Metric(keras.callbacks.Callback):
    """ Keras callback that calculates MAP3 metric. """
    def __init__(self, x_val: NpArray, y_val: NpArray) -> None:
        self.x_val = x_val
        self.y_val = y_val

        self.best_map3 = 0.0
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Any = {}) -> None:
        predict = self.model.predict(x_val)
        map3 = map3_metric(predict, y_val)
        print("epoch %d MAP@3 %.4f" % (epoch, map3))

        if map3 > self.best_map3:
            self.best_map3 = map3
            self.best_epoch = epoch

            self.model.save(get_model_path("val_%.4f" % map3))
            self.model.save(get_best_model_path())

def train_model(x_train: NpArray, x_val: NpArray, y_train: NpArray, y_val:
                NpArray) -> Any:
    """ Creates model and trains it. Returns trained model. """
    x = inp = keras.layers.Input(shape=x_train.shape[1:])
    x = keras.layers.Reshape((x_train.shape[1], x_train.shape[2], 1))(x)

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

    map3 = Map3Metric(x_val, y_val)
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
              verbose=0, validation_data=[x_val, y_val], callbacks=[map3])

    print("best MAP@3 value: %.04f at epoch %d" % (map3.best_map3, map3.best_epoch))
    return model

def predict(x_test: NpArray, label_binarizer: Any) -> NpArray:
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

    if not PREDICT_ONLY:
        train_model(x_train, x_val, y_train, y_val)

    pred = predict(x_test, label_binarizer)

    sub = pd.DataFrame({"fname": test_index, "label": pred})
    sub.to_csv("../submissions/%02x.csv" % CODE_VERSION, index=False, header=True)
    print("submission has been generated")
