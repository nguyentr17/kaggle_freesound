#!/usr/bin/python3.6

import glob, os, sys
from typing import *
from math import log

import numpy as np, pandas as pd, scipy as sp
import keras

from sklearn.model_selection import train_test_split, KFold
from hyperopt import hp, tpe, fmin

from data_v2 import find_files, build_caches, fit_labels
from data_v2 import get_label_binarizer, SoundDatagen


Tensor = Any

TOPK = 3

PREDICT_ONLY    = False
ENABLE_KFOLD    = False
TEST_SIZE       = 0.2
KFOLDS          = 10
USE_HYPEROPT    = True

NUM_CLASSES     = 41
SAMPLE_RATE     = 44100

# Network hyperparameters
NUM_EPOCHS      = 150 if not USE_HYPEROPT else 100


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
    def __init__(self, datagen: SoundDatagen, name: str) -> None:
        self.datagen    = datagen
        self.name       = name
        self.best_map3  = 0.0
        self.best_epoch = 0
        self.last_map3  = 0.0

    def calculate(self) -> float:
        results = []
        count = len(self.datagen)
        generator = iter(self.datagen)

        for _ in range(count):
            x, y = next(generator)
            predict = self.model.predict_on_batch(x)
            map3 = map3_metric(predict, y)
            results.append(map3)

        return np.mean(results)

    def on_epoch_end(self, epoch: int, logs: Any = {}) -> None:
        self.last_map3 = map3 = self.calculate()
        epoch += 1
        print("epoch %d MAP@3 %.4f" % (epoch, map3))

        if map3 > self.best_map3:
            self.best_map3 = map3
            self.best_epoch = epoch

            self.model.save(get_model_path("%s_val_%.4f" % (self.name, map3)))
            self.model.save(get_best_model_path(self.name))

def make_reg(reg: str) -> Any:
    r = float(reg)
    return keras.regularizers.l2(float(10 ** r)) if r >= -5 else None

# def cyclic_lr(epoch: int, current_lr: float) -> float:

def train_model_with_params(params: Dict[str, str], name:str="nofolds") -> float:
    """ Creates model with given parameters. """
    print("training with params", params)

    cnn_dropout_enable  = False # float(params["cnn_dropout_enable"])
    cnn_dropout_coeff   = 0     # float(params["cnn_dropout_coeff"]) * cnn_dropout_enable
    dropout_after_bn    = True  # bool(params["dropout_after_bn"])
    fc_dropout_enable   = True  # float(params["fc_dropout_enable"])
    fc_dropout_coeff    = float(params["fc_dropout_coeff"])
    num_cnn_layers      = int(params["num_cnn_layers"])
    cnn_depth           = int(params["cnn_depth"])
    num_fc_layers       = 1     # int(params["num_fc_layers"])
    num_hidden          = int(params["num_hidden"])
    cnn_kernel_reg      = None    # make_reg(params["cnn_kernel_reg"])
    cnn_bias_reg        = cnn_kernel_reg
    cnn_activity_reg    = cnn_kernel_reg
    fc_kernel_reg       = make_reg("-4.5") # make_reg(params["fc_kernel_reg"])
    fc_bias_reg         = fc_kernel_reg
    fc_activity_reg     = fc_kernel_reg

    padding             = params["padding"]
    cnn_kern_width      = float(params["cnn_kern_width"])
    cnn_kern_height     = float(params["cnn_kern_height"])
    cnn_dim_decay       = float(params["cnn_dim_decay"])
    cnn_depth_growth    = float(params["cnn_depth_growth"])

    shape = train_datagen.shape()
    print("shape of input data", shape)
    x = inp = keras.layers.Input(shape=shape[1:])

    for L in range(num_cnn_layers):
        w = max(int(cnn_kern_width * (cnn_dim_decay ** L)), 2)
        h = max(int(cnn_kern_height * (cnn_dim_decay ** L)), 2)
        d = min(int(cnn_depth * (cnn_depth_growth ** L)), 100)

        # if w >= x._keras_shape[0] or h >= x._keras_shape[1]:
        #     break

        x = keras.layers.Convolution2D(d,
                                       (w, h),
                                       padding=padding,
                                       kernel_regularizer=cnn_kernel_reg,
                                       bias_regularizer=cnn_bias_reg,
                                       activity_regularizer=cnn_activity_reg
                                       )(x)

        if cnn_dropout_coeff and not dropout_after_bn:
            x = keras.layers.Dropout(cnn_dropout_coeff)(x)

        x = keras.layers.BatchNormalization()(x)

        if cnn_dropout_coeff and dropout_after_bn:
            x = keras.layers.Dropout(cnn_dropout_coeff)(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)     # do we want 2x2 pooling?

    x = keras.layers.Flatten()(x)

    for _ in range(num_fc_layers):
        x = keras.layers.Dense(num_hidden,
                               kernel_regularizer=fc_kernel_reg,
                               bias_regularizer=fc_bias_reg,
                               activity_regularizer=fc_activity_reg
                               )(x)

        if fc_dropout_coeff and not dropout_after_bn:
            x = keras.layers.Dropout(fc_dropout_coeff)(x)

        x = keras.layers.BatchNormalization()(x)

        if fc_dropout_coeff and dropout_after_bn:
            x = keras.layers.Dropout(fc_dropout_coeff)(x)

        x = keras.layers.Activation("relu")(x)

    out = keras.layers.Dense(NUM_CLASSES,
                             kernel_regularizer=fc_kernel_reg,
                             bias_regularizer=fc_bias_reg,
                             activity_regularizer=fc_activity_reg,
                             activation="softmax")(x)

    model = keras.models.Model(inputs=inp, outputs=out)
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    map3 = Map3Metric(val_datagen, name)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc',
                        min_delta=0.0001, patience=5, verbose=1)
    # lr_sched = keras.callbacks.LearningRateScheduler(cyclic_lr, verbose=1)

    model.fit_generator(train_datagen, epochs=NUM_EPOCHS,
                        verbose=1, shuffle=False,
                        # use_multiprocessing=False,
                        use_multiprocessing=True, workers=12,
                        validation_data=val_datagen,
                        callbacks=[map3, early_stopping])

    print("best MAP@3 value: %.04f at epoch %d" % (map3.best_map3, map3.best_epoch))
    return map3.last_map3

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

    y_test = model.predict_generator(datagen, verbose=1)
    print("y_test.shape after predict", y_test.shape)

    pos = 0
    y_merged = []

    for count in clips_per_sample:
        if count != 0:
            y_merged.append(merge_predictions(y_test[pos : pos + count],
                                              "max", 0))
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

    if USE_HYPEROPT:
        x_train, x_val, y_train, y_val = train_test_split(
            train_files, train_labels, test_size=TEST_SIZE, shuffle=False)

        train_datagen = SoundDatagen(x_train, y_train)
        val_datagen = SoundDatagen(x_val, y_val)

        '''
        padding             = params["padding"]
        cnn_kern_width      = float(params["cnn_kern_width"])
        cnn_kern_height     = float(params["cnn_kern_height"])
        cnn_dim_decay       = float(params["cnn_dim_decay"])
        cnn_depth_growth    = float(params["cnn_depth_growth"])
        '''

        hyperopt_space = {
            # "cnn_dropout_enable": hp.choice("cnn_dropout_enable", [False, True]),
            # "cnn_dropout_coeff" : hp.uniform("cnn_dropout_coeff", 0.01, 0.99),
            # "dropout_after_bn"  : hp.choice("dropout_after_bn", [False, True]),
            # "fc_dropout_enable" : hp.choice("fc_dropout_enable", [False, True]),
            # "num_fc_layers"     : hp.quniform("num_fc_layers", 0, 2, 1),
            # "cnn_kernel_reg"    : hp.uniform("cnn_kernel_reg", -6, -3),
            # "fc_kernel_reg"     : hp.uniform("fc_kernel_reg", -6, -3),

            "fc_dropout_coeff"  : hp.uniform("fc_dropout_coeff", 0, 0.9),

            "num_cnn_layers"    : hp.quniform("num_cnn_layers", 4, 7, 1),
            "cnn_depth"         : hp.quniform("cnn_depth", 16, 50, 1),
            "num_hidden"        : hp.qloguniform("num_hidden", log(NUM_CLASSES),
                                                 log(100), 1),

            "padding"           : hp.choice("padding", ["same", "valid"]),
            "cnn_kern_width"    : hp.quniform("cnn_kern_width", log(2), log(10), 1),
            "cnn_kern_height"   : hp.quniform("cnn_kern_height", log(2), log(20), 1),
            "cnn_dim_decay"     : hp.choice("cnn_dim_decay", [1, 0.8, 0.75, 0.5]),
            "cnn_depth_growth"  : hp.choice("cnn_depth_growth", [1, 1.25, 1.5, 2]),
        }

        best = fmin(fn=train_model_with_params, space=hyperopt_space,
                    algo=tpe.suggest, max_evals=30)
        print("best params:", best)

        pred = predict(SoundDatagen(test_files, None), "nofolds")
        pred = encode_predictions(pred)
    elif not ENABLE_KFOLD:
        x_train, x_val, y_train, y_val = train_test_split(
            train_files, train_labels, test_size=TEST_SIZE, shuffle=False)

        if not PREDICT_ONLY:
            params : Dict[str, Any] = {
                'cnn_depth': 34.0,
                'cnn_depth_growth': 1.25,
                'cnn_dim_decay': 1,
                'cnn_kern_height': 2.0,
                'cnn_kern_width': 1.0,
                'num_cnn_layers': 6.0,
                'num_hidden': 61.0,
                'padding': 'same'
            }

            train_datagen = SoundDatagen(x_train, y_train)
            val_datagen = SoundDatagen(x_val, y_val)

            train_model_with_params(params)

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
                params = {}

                train_datagen = SoundDatagen(x_train, y_train)
                val_datagen = SoundDatagen(x_val, y_val)

                train_model_with_params(params, name)

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
