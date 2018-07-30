import torch.utils.data as data

from typing import *
import numpy as np, scipy as sp
from PIL import Image

NpArray = Any


class DataGenerator(data.Dataset):
    """ Implements simple data generator. """
    def __init__(self, x: NpArray, y: NpArray, transform: Any = None,
                 datagen: Any = None) -> None:
        print("x", x.shape)

        self.x = x
        self.y = y
        self.sample_num = x.shape[0]
        self.datagen = datagen
        self.transform = transform

    def __len__(self) -> int:
        return self.sample_num

    def __getitem__(self, i: int) -> NpArray:
        x = self.x[i]

        if self.datagen:
            x[i] = self.datagen.random_transform(x[i])

        y = self.y[i]
        x = Image.fromarray(x, "F")

        if self.transform:
            x = self.transform(x)

        x = x.expand(3, -1, -1)
        return x, y
