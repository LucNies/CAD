# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:14:52 2016

@author: Luc
"""
from __future__ import division
from random import shuffle
import numpy as np
import patcher
import load_data


def get_features_labels(images, truth=None, patching=patcher.ImPatch()):
    features = []
    labels = []
    
    #alleen voor batchsize 1!!
    patches, coords = patching.patch(images)
    features = patches#aanpassen als we vaker willen
    
    shuffled = np.arange(len(features))
    shuffle(shuffled)
    features = reorder(features, shuffled)

    if truth is not None:
        for (x,y) in coords:
            if(len(truth.shape)==3):
                labels.append(truth[0][x][y])
            else:
                labels.append(truth[x][y])
        labels = reorder(labels, shuffled)
        return features, labels
    
    return features


def remove_threshold(patches, labels, coords, t = 1):
    toRemove = []
    for i,patch in enumerate(patches):
        if patch[0] < t: #find the correct indices of the middle pixel of al three channels
            toRemove.append(i)
    
    return np.delete(patches, toRemove), np.delete(labels, toRemove), np.delete(labels, toRemove)


def calc_dice(predictions, labels):
    NNEQ = np.sum(predictions != labels)
    NNT = np.sum(predictions == labels)
    NNZ = np.sum((predictions + labels) != 0)
    return NNEQ/float(NNT+NNZ)

    
def reorder(array, order):
    result = [array[i] for i in order]
    return result
            
def create_location_feature():
        loader = load_data.loader()
        data, truth = loader.load_batch()
        locations = np.zeros()

