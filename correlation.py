#!/usr/bin/python3.6
import sys
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: %s prediction1.npz prediction2.npz")
        sys.exit()

    print("reading", sys.argv[1])
    pred1 = np.load(sys.argv[1])["predict"]

    print("reading", sys.argv[2])
    pred2 = np.load(sys.argv[2])["predict"]

    print("RMS of difference:", np.sqrt(np.mean((pred1 - pred2) ** 2)))
    print("standard deviation:", np.std(pred1 - pred2))
    print("correlation:", np.corrcoef(pred1, pred2)[0, 1])
