import torch.utils.data as data

from typing import *
import numpy as np
NpArray = Any


class MixupGenerator(data.Dataset):
    """ Implements mixup of audio clips. """
    def __init__(self, X_train: NpArray, y_train: NpArray, transform: Any,
                 batch_size: int = 32, alpha: float = 0.2, shuffle: bool = True,
                 datagen: Any = None) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen
        self.indexes = self.__get_exploration_order()
        self.transform = transform

    def __get_exploration_order(self) -> NpArray:
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __len__(self) -> int:
        return int(len(self.indexes) // (self.batch_size * 2))

    def __getitem__(self, i: int) -> np.array:
        batch_ids = self.indexes[i * self.batch_size * 2 :
                                 (i + 1) * self.batch_size * 2]
        h, w = self.X_train.shape[1], self.X_train.shape[2]

        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        if self.transform:
            X = self.transform(X)
            
        return X, y
