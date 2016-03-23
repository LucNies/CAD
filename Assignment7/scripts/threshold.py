import skimage as ski
from skimage import exposure
import os
import numpy as np
import skimage.io as sio
from features import calc_dice
from classify import CLF

class Optimizer:

    def __init__(self):
        self.all_labels = []
        self.all_data = []

    def partial_fit(self, data, labels, _):
        n = np.sqrt(len(data[0])/3)
        m = int(n/2)+1
        for l,d in zip(labels, data):
            self.all_labels.append(l)

            px = d[m*n-m]
            self.all_data.append(px)

    def predict_proba(self, data):
        start, stop, step = 0,1,0.1

        scores = []
        for t in np.arange(start, stop, step):
            scores.append(self.score(self.all_data, t))

        opt_t = start + np.argmax(scores) * step

        px = np.zeros((len(data),))
        n = np.sqrt(len(data[0])/3)
        m = int(n/2)+1
        for i, d in enumerate(data):
            px[i] = d[m*n-m]
        return np.array([(1-p /255., p /255.) for p in px])

    def score(self, px, t):
        pred = np.array([p > t for p in px])
        return calc_dice(pred, self.all_labels)


if __name__ == "__main__":
    pass
