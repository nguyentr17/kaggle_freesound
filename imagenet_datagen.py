from typing import *
import numpy as np, scipy as sp
from PIL import Image
import torch.utils.data as data

NpArray = Any


class DataGenerator(data.Dataset):
    """ Implements simple data generator. """
    def __init__(self, x: NpArray, y: NpArray, transform: Any = None) -> None:
        print("DataGenerator: converting data, x", x.shape)

        self.sample_num = x.shape[0]
        self.transform = transform

        self.x = [self.convert_image(img) for img in x]
        self.y = y

    def __len__(self) -> int:
        return self.sample_num

    def convert_image(self, img: NpArray) -> NpArray:
        img = Image.fromarray(img, "F")

        if self.transform:
            img = self.transform(img)

        img = img.expand(3, -1, -1)
        return img

    def __getitem__(self, i: int) -> NpArray:
        return self.x[i], self.y[i]
