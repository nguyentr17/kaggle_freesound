#!/usr/bin/python3.6

import glob
import matplotlib.pyplot as plt
import librosa

if __name__ == "__main__":
    train_dataset = list(glob.glob("../data/audio_train/*.wav"))
    print("train_dataset: %d files found" % len(train_dataset))

    for train_sample in train_dataset[:1]:
        print("reading file '%s'" % train_sample)

        wave, sr = librosa.core.load(train_sample, sr=None)
        assert(sr == 44100)
        print("sound data shape:", wave.shape)

        mfcc = librosa.feature.mfcc(wave, sr)
        print("mfcc shape:", mfcc.shape)

        plt.imshow(mfcc)
        plt.show()
