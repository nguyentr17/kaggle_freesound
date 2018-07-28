#!/usr/bin/python3.6

import glob, os, pickle, sys
from typing import *

import numpy as np, pandas as pd, scipy as sp
import keras
from hyperopt import hp, tpe, fmin

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight

from data import load_dataset
from data_v1_1 import get_random_eraser, MixupGenerator, AugGenerator


NpArray = Any

TOPK = 3

CODE_VERSION    = os.path.splitext(os.path.basename(__file__))[0][1:]
DATA_VERSION    = 0x01

PREDICT_ONLY    = False
ENABLE_KFOLD    = False
ENABLE_HYPEROPT = True
TEST_SIZE       = 0.2
KFOLDS          = 20

NUM_CLASSES     = 41
SAMPLE_RATE     = 44100

# Network hyperparameters
BATCH_SIZE      = 32
NUM_EPOCHS      = 50

HYPEROPT_EVALS  = 25


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

    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)

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
    """ Creates model and trains it. Returns MAP@3 metric. """
    cnn_dropout     = float(params["cnn_dropout"])
    dropout_coeff   = float(params["dropout_coeff"])
    reg_coeff       = float(params["reg_coeff"])
    reg_coeff2      = reg_coeff
    conv_depth      = 27 # int(params["conv_depth"])
    num_hidden      = int(params["num_hidden"])
    conv_width      = 4 # int(params["conv_width"])
    conv_height     = 6 # int(params["conv_height"])
    conv_count      = 4 # int(params["conv_count"])

    # TODO: search for the best pooling size

    shift           = float(params["shift"])
    flip            = False # bool(params["flip"])
    erase           = bool(params["erase"])
    alpha           = float(params["alpha"])
    mixup           = True # bool(params["mixup"])

    print("training with params", params)

    reg = keras.regularizers.l2(10 ** reg_coeff)
    reg2 = keras.regularizers.l2(10 ** reg_coeff2)

    x = inp = keras.layers.Input(shape=x_train.shape[1:])
    x = keras.layers.Reshape((x_train.shape[1], x_train.shape[2], 1))(x)

    for _ in range(conv_count):
        x = keras.layers.Convolution2D(conv_depth, (conv_width, conv_height),
                                       padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)

        if cnn_dropout > 0.001:
            x = keras.layers.Dropout(cnn_dropout)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(num_hidden,
                           kernel_regularizer=reg,
                           bias_regularizer=reg,
                           activity_regularizer=reg,
                           )(x)

    if dropout_coeff > 0.001:
        x = keras.layers.Dropout(dropout_coeff)(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    out = keras.layers.Dense(NUM_CLASSES,
                             kernel_regularizer=reg2,
                             bias_regularizer=reg2,
                             activity_regularizer=reg2,
                             activation="softmax")(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    map3 = Map3Metric(x_val, y_val, name)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                    factor=0.33, patience=7, verbose=1, min_lr=3e-6)

    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=shift, # 0.4,  # randomly shift images horizontally (fraction of total width)
        horizontal_flip=flip,   # randomly flip images
        preprocessing_function=
            # random eraser
            get_random_eraser(v_l=np.min(x_train), v_h=np.max(x_train)) if erase else None
    )

    if mixup:
        mixupgen = MixupGenerator(x_train, y_train, alpha=alpha,
                                  batch_size=BATCH_SIZE, datagen=datagen)
    else:
        mixupgen = AugGenerator(x_train, y_train, alpha=alpha,
                                batch_size=BATCH_SIZE, datagen=datagen)

    model.fit_generator(mixupgen,
                        epochs=NUM_EPOCHS, verbose=1,
                        validation_data=[x_val, y_val],
                        use_multiprocessing=True, workers=12,
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
        # return sp.stats.gmean(pred, axis=axis)
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

    train_labels = stratify=train_df["label"]
    class_weight = class_weight.compute_class_weight('balanced',
                       np.unique(train_labels), train_labels)
    class_weight = {i: w for i, w in enumerate(class_weight)}
    print(class_weight)

    if ENABLE_HYPEROPT:
        train_idx, val_idx = train_test_split(train_indices,
                                              stratify=train_labels,
                                              random_state=0,
                                              test_size=TEST_SIZE)
        x_train, y_train, x_val, y_val, x_test, label_binarizer, \
            clips_per_sample = load_data(train_idx, val_idx)

        '''
        cnn_dropout     = float(params["cnn_dropout"])
        dropout_coeff   = float(params["dropout_coeff"])
        reg_coeff       = float(params["reg_coeff"])
        num_hidden      = int(params["num_hidden"])
        shift           = float(params["shift"])
        erase           = bool(params["erase"])
        alpha           = float(params["alpha"])
        '''

        hyperopt_space = {
            "cnn_dropout"       : hp.uniform("cnn_dropout", 0, 0.2),
            "dropout_coeff"     : hp.uniform("dropout_coeff", 0.3, 0.6),
            "reg_coeff"         : hp.uniform("reg_coeff", -10, -4),
            "num_hidden"        : hp.quniform("num_hidden", 32, 128, 1),
            "shift"             : hp.uniform("shift", 0, 0.5),
            "erase"             : hp.choice("erase", [True, False]),
            "alpha"             : hp.uniform("alpha", 0.001, 0.999),
        }

        best = fmin(fn=train_model, space=hyperopt_space,
                    algo=tpe.suggest, max_evals=HYPEROPT_EVALS)
        print("best params:", best)

        pred = predict(x_test, label_binarizer, clips_per_sample, "nofolds")
        pred = encode_predictions(pred)
    elif not ENABLE_KFOLD:
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
        pred = encode_predictions(pred)
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
        pred = encode_predictions(pred)
        print("predictions after encoding", pred.shape)

    sub = pd.DataFrame({"fname": test_idx, "label": pred})
    sub.to_csv("../submissions/%s.csv" % CODE_VERSION, index=False, header=True)
    print("submission has been generated")
