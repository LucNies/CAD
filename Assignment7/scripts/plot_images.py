# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:15:19 2016

@author: Luc
"""
import numpy as np
import matplotlib.pyplot as plt

def plot(prediction, label):
    im = np.hstack((prediction, label))
    plt.figure();
    plt.imshow(im, cmp = "Greys_r")
    plt.show();