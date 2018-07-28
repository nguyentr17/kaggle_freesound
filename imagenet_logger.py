import logging
from typing import *

import numpy as np

def create_logger(filename: str, logger_name: str='logger',
                  file_fmt: Any ='%(asctime)s %(levelname)-8s: %(message)s',
                  console_fmt: Any ='%(message)s',
                  file_level: int = logging.DEBUG, console_level: int = logging.DEBUG) -> Any:

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_fmt = logging.Formatter(file_fmt)
    log_file = logging.FileHandler(filename)
    log_file.setLevel(file_level)
    log_file.setFormatter(file_fmt)
    logger.addHandler(log_file)

    console_fmt = logging.Formatter(console_fmt)
    log_console = logging.StreamHandler()
    log_console.setLevel(logging.DEBUG)
    log_console.setFormatter(console_fmt)
    logger.addHandler(log_console)

    return logger

class AverageMeter(object):
    """ Computes and stores the average and current value. """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def rmse(output: Any, target: Any) -> float:
#     """ Computes the metric. """
#     import torch
#     return (torch.sum((output - target) ** 2).item() / target.size(0)) ** 0.5
