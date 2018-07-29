import torch.utils.data as data

from typing import *
import numpy as np, scipy as sp
NpArray = Any


class MixupGenerator(data.Dataset):
    """ Implements mixup of audio clips. """
    def __init__(self, x_train: NpArray, y_train: NpArray, transform: Any = None,
                 batch_size: int = 32, alpha: float = 0.2, shuffle: bool = True,
                 datagen: Any = None) -> None:
        print("x", x_train.shape)
        x_train = np.expand_dims(x_train, axis=1)
        x_train = np.repeat(x_train, 3, axis=1)
        print("x after repeating", x_train.shape)

        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = x_train.shape[0]
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
        h, w = self.x_train.shape[1], self.x_train.shape[2]

        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        x_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        x1 = self.x_train[batch_ids[:self.batch_size]]
        x2 = self.x_train[batch_ids[self.batch_size:]]
        x = x1 * x_l + x2 * (1 - x_l)

        if self.datagen:
            for i in range(self.batch_size):
                x[i] = self.datagen.random_transform(x[i])

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
            x = self.transform(x)
        else:
            x = np.array([sp.misc.imresize(img, 2.0) for img in x])
            x = np.transpose(x, (0, 3, 1, 2))

        print("returning x, y", x.shape, y.shape) # type: ignore
        return x, y
