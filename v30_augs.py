#!/usr/bin/python3.6

import glob, os, pickle, sys, time
from typing import *
from math import log

import numpy as np, pandas as pd, scipy as sp
import keras

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from hyperopt import hp, tpe, fmin

from data_v1_1 import load_dataset, DATA_VERSION
from data_v1_1 import get_random_eraser, MixupGenerator


NpArray = Any
Tensor = Any

TOPK = 3

CODE_VERSION    = os.path.splitext(os.path.basename(__file__))[0]

PREDICT_ONLY    = False
ENABLE_KFOLD    = False
TEST_SIZE       = 0.2
KFOLDS          = 10
USE_HYPEROPT    = True

NUM_CLASSES     = 41
SAMPLE_RATE     = 44100
MAX_MFCC        = 40

# Network hyperparameters
NUM_EPOCHS      = 50


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
    """ Loads all data. Returns: x_train, y_train, x_val, y_val, x_test,
    label_binarizer, clips_per_sample """
    train_df = pd.read_csv("../data/train.csv", index_col="fname")

    train_cache = "../output/train_cache_v%02x.pkl" % DATA_VERSION
    test_cache = "../output/test_cache_v%02x.pkl" % DATA_VERSION

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

    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)

    x_train = x_train[:, :MAX_MFCC, ...]
    x_val = x_val[:, :MAX_MFCC, ...]
    x_test = x_test[:, :MAX_MFCC, ...]

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

def make_reg(reg: float) -> Any:
    return keras.regularizers.l2(10 ** reg) if reg >= -6 else None

class TimedStopping(keras.callbacks.Callback):
    """ Stop training when enough time has passed. """
    def __init__(self, timeout: float, verbose: int = 0) -> None:
        super(keras.callbacks.Callback, self).__init__()

        self.start_time = 0.0
        self.seconds = timeout
        self.verbose = verbose

    def on_train_begin(self, logs: Any = {}) -> None:
        self.start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Any = {}) -> None:
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose:
                print("stopping after %s seconds." % self.seconds)

class Map3Metric(keras.callbacks.Callback):
    """ Keras callback that calculates MAP3 metric. """
    def __init__(self, x_val: NpArray, y_val: NpArray, name: str) -> None:
        self.x_val = x_val
        self.y_val = y_val
        self.name = name

        self.best_map3 = 0.0
        self.best_epoch = 0
        self.last_map3  = 0.0

        self.last_best_map3 = 0.0
        self.last_best_epoch = 0
        self.max_epochs = 5
        self.min_threshold = 0.01

    def on_epoch_end(self, epoch: int, logs: Any = {}) -> None:
        predict = self.model.predict(self.x_val)
        self.last_map3 = map3 = map3_metric(predict, self.y_val)
        print("epoch %d MAP@3 %.4f" % (epoch, map3))

        if map3 > self.best_map3:
            self.best_map3 = map3
            self.best_epoch = epoch

            self.model.save(get_model_path("%s_val_%.4f" % (self.name, map3)))
            self.model.save(get_best_model_path(self.name))

        # Optionally do early stopping basing on MAP@3 metric
        if self.best_map3 > self.last_best_map3 + self.min_threshold:
            self.last_best_map3 = map3
            self.last_best_epoch = epoch
        elif epoch >= self.last_best_epoch + self.max_epochs:
            self.model.stop_training = True
            print("stopping training because MAP@3 growth has stopped")

lr_cycle_len        = 4
min_lr              = 10 ** -5.5
max_lr              = 10 ** -4.5

def cyclic_lr(epoch: int) -> float:
    effective_max_lr = max_lr
    if epoch != 0 and epoch % 20 == 0:
        effective_max_lr *= 0.1

    cycle = np.floor(1 + epoch / (2 * lr_cycle_len))
    x = abs(epoch / lr_cycle_len - 2 * cycle + 1)
    lr = min_lr + (effective_max_lr - min_lr) * min(1, x)
    return lr

def train_model_with_params(params: Dict[str, str], name:str="nofolds") -> float:
    """ Creates model with given parameters. """
    print("training with params", params)

    batch_size      = int(params["batch_size"])
    num_hidden      = int(params["num_hidden"])
    num_fc_layers   = int(params["num_fc_layers"])
    num_conv_layers = int(params["num_conv_layers"])
    conv_depth      = int(params["conv_depth"])
    conv_depth_mul  = float(params["conv_depth_mul"])
    conv_width      = int(params["conv_width"])
    conv_height     = int(params["conv_height"])
    pooling_size    = int(params["pooling_size"])
    pooling_type    = params["pooling_type"]
    dropout_coeff   = float(params["dropout_coeff"])
    reg_coeff       = float(params["reg_coeff"])
    residuals       = "" # params["residuals"]

    shift           = float(params["shift"])
    flip            = bool(params["flip"])
    erase           = False # bool(params["erase"])
    alpha           = float(params["alpha"])

    if residuals is not "":
        conv_depth_mul = 1

    fc_bias_reg = fc_activity_reg = fc_kernel_reg = make_reg(reg_coeff)

    print("x_train:", x_train.shape)
    x = inp = keras.layers.Input(shape=x_train.shape[1:])
    # x = keras.layers.Reshape((x_train.shape[1], x_train.shape[2], 1))(x)

    w, h = x_train.shape[1], x_train.shape[2]
    conv_layers = []

    for L in range(num_conv_layers):
        if pooling_type.startswith("local"):
            w, h = (w + 1) / pooling_size, (h + 1) / pooling_size
            if w < 2 or h < 2:
                break

        conv_layers.append(x)
        if L % 2 == 0:
            prev = x

        d = min(int(conv_depth * (conv_depth_mul ** L)), 100)
        x = keras.layers.Convolution2D(d, (conv_width, conv_height),
                                       padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        if pooling_type.startswith("global"):
            if residuals == "resnet" and L % 2 == 1:
                x = keras.layers.Add()([prev, x])
            elif residuals == "densenet":
                # print("conv_layers", conv_layers)
                x = keras.layers.Add()(conv_layers + [x])

        x = keras.layers.Activation("relu")(x)

        if pooling_type == "local_max":
            x = keras.layers.MaxPooling2D((pooling_size, pooling_size))(x)
        elif pooling_type == "local_avg":
            x = keras.layers.AveragePooling2D((pooling_size, pooling_size))(x)

    if pooling_type == "global_max":
        x = keras.layers.GlobalMaxPooling2D()(x)
    elif pooling_type == "global_avg":
        x = keras.layers.GlobalAveragePooling2D()(x)
    else:
        x = keras.layers.Flatten()(x)

    for _ in range(num_fc_layers):
        x = keras.layers.Dense(num_hidden,
                               kernel_regularizer=fc_kernel_reg,
                               bias_regularizer=fc_bias_reg,
                               activity_regularizer=fc_activity_reg
                               )(x)
        x = keras.layers.Dropout(dropout_coeff)(x)
        x = keras.layers.BatchNormalization()(x)
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

    if model.count_params() >= 4 * 10 ** 6:
        return 0.5

    map3 = Map3Metric(x_val, y_val, name)
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
    #                     min_delta=0.01, patience=10, verbose=1)
    lr_shed = keras.callbacks.LearningRateScheduler(cyclic_lr, verbose=1)
    time_stopping = TimedStopping(timeout=15*60*60, verbose=1)

    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=shift, # 0.4,  # randomly shift images horizontally (fraction of total width)
        horizontal_flip=flip,   # randomly flip images
        preprocessing_function=
            get_random_eraser(v_l=np.min(x_train), v_h=np.max(x_train)) if erase else None # random eraser
    )
    mixupgen = MixupGenerator(x_train, y_train, alpha=alpha, batch_size=batch_size,
                              datagen=datagen)

    model.fit_generator(mixupgen,
                        epochs=NUM_EPOCHS, verbose=1,
                        validation_data=[x_val, y_val],
                        use_multiprocessing=True, workers=12,
                        callbacks=[map3, lr_shed, time_stopping])

    print("best MAP@3 value: %.04f at epoch %d" % (map3.best_map3, map3.best_epoch))
    return map3.last_map3

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
    train_df = pd.read_csv("../data/train.csv", index_col="fname")
    train_indices = range(train_df.shape[0])

    test_files = find_files("../data/audio_test/")
    test_idx = [os.path.basename(f) for f in test_files]

    if USE_HYPEROPT:
        train_idx, val_idx = train_test_split(train_indices, shuffle=False,
                                              test_size=TEST_SIZE)
        x_train, y_train, x_val, y_val, x_test, label_binarizer, \
            clips_per_sample = load_data(train_idx, val_idx)


        '''
        batch_size      = int(params["batch_size"])
        num_hidden      = int(params["num_hidden"])
        num_fc_layers   = int(params["num_fc_layers"])
        num_conv_layers = int(params["num_conv_layers"])
        conv_depth      = int(params["conv_depth"])
        conv_depth_mul  = int(params["conv_depth"])
        conv_width      = int(params["conv_width"])
        conv_height     = int(params["conv_height"])
        pooling_size    = int(params["pooling_size"])
        pooling_type    = params["pooling_type"]
        dropout_coeff   = float(params["dropout_coeff"])
        reg_coeff       = float(params["reg_coeff"])
        residuals       = params["residuals"]

        shift           = params["shift"]
        flip            = params["flip"]
        erase           = params["erase"]
        alpha           = params["alpha"]
        '''

        hyperopt_space = {
            "batch_size"        : hp.choice("batch_size", [16, 32, 64]),
            "num_hidden"        : hp.qloguniform("num_hidden", log(41), log(100), 1),
            "num_fc_layers"     : hp.choice("num_fc_layers", [0, 1, 2]),
            "num_conv_layers"   : hp.quniform("num_conv_layers", 4, 10, 1),
            "conv_depth"        : hp.quniform("conv_depth", 16, 50, 1),
            "conv_depth_mul"    : hp.loguniform("conv_depth_mul", log(1), log(2)),
            "conv_width"        : hp.quniform("conv_width", 1, 10, 1),
            "conv_height"       : hp.quniform("conv_height", 1, 20, 1),
            "pooling_size"      : hp.choice("pooling_size", [2, 3]),
            "pooling_type"      : hp.choice("pooling_type", ["local_max", "local_avg",
                                                             "global_max", "global_avg"]),
            "dropout_coeff"     : hp.uniform("dropout_coeff", 0.4, 0.6),
            "reg_coeff"         : hp.uniform("reg_coeff", -5, -3),
            # "residuals"         : hp.choice("residuals", ["resnet", "densenet", ""]),

            "shift"             : hp.uniform("shift", 0, 0.5),
            "flip"              : hp.choice("flip", [True, False]),
            # "erase"             : hp.choice("erase", [True, False]),
            "alpha"             : hp.uniform("alpha", 0.001, 0.999),
        }

        best = fmin(fn=train_model_with_params, space=hyperopt_space,
                    algo=tpe.suggest, max_evals=200)
        print("best params:", best)

        pred = predict(x_test, label_binarizer, clips_per_sample, "nofolds")
    elif not ENABLE_KFOLD:
        train_idx, val_idx = train_test_split(train_indices, shuffle=False,
                                              test_size=TEST_SIZE)
        x_train, y_train, x_val, y_val, x_test, label_binarizer, \
            clips_per_sample = load_data(train_idx, val_idx)

        if not PREDICT_ONLY:
            params : Dict[str, Any] = {
                # some random parameters, don't expect much
                'batch_size': 16,
                'conv_depth': 18.0,
                'conv_depth_mul': 1.547410573796067,
                'conv_height': 14.0,
                'conv_width': 8.0,
                'dropout_coeff': 0.4942919219184629,
                'num_conv_layers': 10.0,
                'num_fc_layers': 1,
                'num_hidden': 43.0,
                'pooling_size': 3,
                'pooling_type': 'global_avg',
                'reg_coeff': -4.853239812417038,
                'residuals': ''
             }

            train_model_with_params(params)

        pred = predict(x_test, label_binarizer, clips_per_sample, "nofolds")
    else:
        kf = KFold(n_splits=KFOLDS, shuffle=False)
        pred = np.zeros((len(test_idx), KFOLDS, NUM_CLASSES))

        for k, (train_idx, val_idx) in enumerate(kf.split(train_indices)):
            print("fold %d ==============================================" % k)

            x_train, y_train, x_val, y_val, x_test, label_binarizer, \
                clips_per_sample = load_data(train_idx, val_idx)
            name = "fold_%d" % k

            if not PREDICT_ONLY:
                params = {
                }

                train_model_with_params(params, name)

            pred[:, k, :] = predict(x_test, label_binarizer, clips_per_sample, name)

        print("before final merge: pred.shape", pred.shape)
        pred = merge_predictions(pred, "geom_mean", axis=1)
        print("predictions after final merge", pred.shape)

    np.savez("../predictions/%s.npz" % CODE_VERSION, predict=pred)
    print("matrix of predictions has been saved")
