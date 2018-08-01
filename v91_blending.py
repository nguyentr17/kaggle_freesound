#!/usr/bin/python3.6

import glob, pickle, os, sys
from typing import *

import numpy as np, pandas as pd
from scipy import stats
import keras

# from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer

from data_v1 import load_everything, load_data, map3_metric
import data_v4


CODE_VERSION    = os.path.splitext(os.path.basename(__file__))[0][1:]

NpArray         = Any
TOPK            = 3

PREDICT_ONLY    = False
ENABLE_KFOLD    = True
TEST_SIZE       = 0.2
KFOLDS          = 7

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


# class Map3Metric(keras.callbacks.Callback):
#     """ Keras callback that calculates MAP3 metric. """
#     def __init__(self, x_val: NpArray, y_val: NpArray, name: str) -> None:
#         self.x_val = x_val
#         self.y_val = y_val
#         self.name = name
#
#         self.best_map3 = 0.0
#         self.best_epoch = 0
#
#         self.last_best_map3 = 0.0
#         self.last_best_epoch = 0
#         self.max_epochs = 15
#         self.min_threshold = 0.001
#
#     def on_epoch_end(self, epoch: int, logs: Any = {}) -> None:
#         predict = self.model.predict(self.x_val)
#         map3 = map3_metric(predict, self.y_val)
#         print("epoch %d MAP@3 %.4f" % (epoch + 1, map3))
#
#         if map3 > self.best_map3:
#             self.best_map3 = map3
#             self.best_epoch = epoch
#
#             self.model.save(get_model_path("%s_val_%.4f" % (self.name, map3)))
#             self.model.save(get_best_model_path(self.name))
#
#         # Optionally do early stopping, basing on MAP@3 metric
#         if self.best_map3 > self.last_best_map3 + self.min_threshold:
#             self.last_best_map3 = map3
#             self.last_best_epoch = epoch
#         elif epoch >= self.last_best_epoch + self.max_epochs:
#             self.model.stop_training = True
#             print("stopping training because MAP@3 growth has stopped")
#
# def train_model(name: str ="nofolds") -> float:
#     """ Creates model, trains it and saves it to the file. """
#     shape = x_train.shape
#     x = inp = keras.layers.Input(shape=shape[1:])
#
#     reg = None # keras.regularizers.l2(10 ** 4.5)
#     x = keras.layers.Dense(128,
#                            kernel_regularizer=reg,
#                            bias_regularizer=reg,
#                            activity_regularizer=reg,
#                            )(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#
#     out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
#
#     model = keras.models.Model(inputs=inp, outputs=out)
#     model.summary()
#
#     model.compile(loss="categorical_crossentropy", optimizer="adam",
#                   metrics=["accuracy"])
#
#     map3 = Map3Metric(x_val, y_val, name)
#     reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
#                     factor=0.33, patience=7, verbose=1, min_lr=3e-6)
#
#     model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
#               verbose=1, validation_data=[x_val, y_val],
#               callbacks=[map3, reduce_lr])
#
#     print("best MAP@3 value: %.04f at epoch %d" % (map3.best_map3,
#                                                    map3.best_epoch + 1))
#     return map3.best_map3

# def merge_predictions(pred: NpArray, mode: str, axis: int) -> NpArray:
#     """ Merges predictions for all clips and returns a single prediction. """
#     assert(pred.shape[-1] == NUM_CLASSES)
#
#     if mode == "mean":
#         return np.mean(pred, axis=axis)
#     elif mode == "max":
#         print("pred.shape", pred.shape)
#         return np.max(pred, axis=axis)
#     elif mode == "geom_mean":
#         # this code is same as in scipy, but it prevents warning
#         res = 1
#
#         for i in range(pred.shape[axis]):
#             res = res * np.take(pred, i, axis=axis)
#
#         return res ** (1 / pred.shape[axis])
#     else:
#         assert(False)

def predict(x_test: NpArray, model_path: str) -> NpArray:
    """ Predicts results on test, using the best model. """
    print(f"predicting results, x.shape={x_test.shape}")
    # model = keras.models.load_model(get_best_model_path(model_name))
    model = keras.models.load_model(model_path)

    y_test = model.predict(x_test, verbose=1)
    print(f"y_test.shape after predict: {y_test.shape}")
    return y_test

# def predict_and_merge(x_test: NpArray, label_binarizer: Any,
#                       clips_per_sample: List[int], model_path: str) -> NpArray:
#     """ Predicts results on test, using the best model. """
#     y_test = predict(x_test, model_path)
#
#     pos = 0
#     y_merged = []
#
#     for count in clips_per_sample:
#         if count != 0 and pos < y_test.shape[0]:
#             # print(f"pos={pos} pos+count={pos+count}", y_test.shape, y_test[pos : pos+count].shape)
#             y_merged.append(merge_predictions(y_test[pos : pos+count], "max", 0))
#         else:
#             y_merged.append(np.zeros_like(y_merged[0]))
#
#         pos += count
#
#     y_test = np.array(y_merged)
#     print("y_test.shape after merge", y_test.shape)
#     return y_test

def out_of_fold_predict(model_name: str) -> NpArray:
    """ Predicts result on train in OOF way. """
    folds = list(glob.glob(f"../models/{model_name}__fold_*_best.hdf5"))
    k_folds = len(folds)
    print(f"predicting model={model_name}, k_folds={k_folds}")
    assert(k_folds == 10 or k_folds == 20 or k_folds == 7)

    cache_path = f"../output/predict_{model_name}.pkl"
    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))
    else:
        kf = StratifiedKFold(n_splits=k_folds, shuffle=False)
        pred = np.zeros((len(train_indices), NUM_CLASSES))

        for k, (train_idx, val_idx) in enumerate(kf.split(train_indices,
                                                          train_labels)):
            print(f"predicting fold {k}")

            if model_name.startswith("71"):
                x_train, y_train, x_val, y_val, x_test, label_binarizer, \
                    clips_per_sample = data_v4.load_data(train_idx, val_idx)
            else:
                x_train, y_train, x_val, y_val, x_test, label_binarizer, \
                    clips_per_sample = load_data(train_idx, val_idx)

            p = predict(x_val, folds[k])

            for i, idx in enumerate(val_idx):
                pred[idx, :] = p[i]

        pickle.dump(pred, open(cache_path, "wb"))
        return pred

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
    train_indices, test_idx, train_labels, class_weights = load_everything()

    v55 = out_of_fold_predict("55")
    print(f"v55: {v55.shape}")
    v71 = out_of_fold_predict("71_more_folds")
    print(f"v71: {v71.shape}")
    v80 = out_of_fold_predict("80_lstm")
    print(f"v80: {v80.shape}")
    v85 = out_of_fold_predict("85_gru")
    print(f"v85: {v85.shape}")
    v86 = out_of_fold_predict("86_gru")
    print(f"v86: {v86.shape}")

    oof_train_x = np.concatenate([v55, v71, v80, v86], axis=1)
    print(f"oof data shape: {oof_train_x.shape}")

    v55_test = np.load("../predictions/55_balanced.npz")["predict"]
    print(f"v55_test: {v55_test.shape}")
    v71_test = np.load("../predictions/71_more_folds.npz")["predict"]
    v80_test = np.load("../predictions/80_lstm.npz")["predict"]
    v85_test = np.load("../predictions/85_gru.npz")["predict"]
    v86_test = np.load("../predictions/86_gru.npz")["predict"]

    kf = StratifiedKFold(n_splits=KFOLDS, shuffle=False)
    pred = np.zeros((len(test_idx), KFOLDS, NUM_CLASSES))

    train_df = pd.read_csv("../data/train.csv", index_col="fname")
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(train_df["label"])
    oof_train_y = label_binarizer.transform(train_df["label"])
    print(f"oof_train_y: {oof_train_y.shape}")


    ### finding the optimum weights

    predictions = [v55, v71, v80, v85, v86]

    def log_loss_func(weights: NpArray) -> float:
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0

        for weight, prediction in zip(weights, predictions):
                final_prediction += weight * prediction

        return log_loss(oof_train_y, final_prediction)

    # the algorithm needs a starting value, right now we chose 0.5 for all weights
    # it's better to choose many random starting points and run minimize a few times
    starting_values = [0.5] * len(predictions)

    # adding constraints  and a different solver as suggested by user 16universe
    # https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    # our weights are bound between 0 and 1
    bounds = [(0, 1)] * len(predictions)

    res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

    print(f"ensemble score: {res['fun']}")
    print(f"best weights: {res['x']}")


    weights = res["x"]
    predictions = [v55_test, v71_test, v80_test, v85_test, v86_test]
    final_prediction = 0

    for weight, prediction in zip(weights, predictions):
        final_prediction += weight * prediction

    pred = encode_predictions(final_prediction)
    print("predictions after encoding", pred.shape)

    print("final_prediction", pred.shape) # type: ignore
    sub = pd.DataFrame({"fname": test_idx, "label": pred})
    sub.to_csv("../submissions/auto_blend.csv", index=False, header=True)
    print("submission has been generated")
