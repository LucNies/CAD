import skimage as ski
from skimage import exposure
import os
import numpy as np
import skimage.io as sio
from features import calc_dice
from classify import CLF

class Optimizer:

    def __init__(self, test=None, search_space=None):
        """Makes the optimiser.
        :param test: function that takes threshold and outputs a performance value between 0 and 1
        :param search_space: triple with start, stop and step values for range of thresholds
        :return: Optimizer object
        """
        if (test is None):
            self.test = lambda x: 0.5
        else:
            self.test = test
        if (search_space is None):
            self.search_space = (0,1,0.1)
        else:
            self.search_space = search_space

    def loadimg(self, imgnm):
        img = sio.imread(imgnm)
        img = exposure.equalize_hist(img)
        img = (img * 255).astype(int)
        return img

    def optimize(self, dir):
        print "Optimizing threshold"
        """Main function that finds the best threshold
        """
        start, stop, step = self.search_space

        #imgnames = [os.path.join(dir,nm) for nm in os.listdir(dir) if nm[-6:]=="fl.png"]

        t, p = self.test()
        print "Lowest error: {}@t={}".format(p,t)

if __name__ == "__main__":
    sgd = CLF()
    sgd.train()
    O = Optimizer(sgd.test)
    t = O.optimize("../data")
    sio.imshow(O.loadimg("../data/1230931003_fl.png")>t)
    truth = sio.imread("../data/1230931003_an.png")
    #print np.sum(truth) / np.prod(np.shape(truth)) / 255.
