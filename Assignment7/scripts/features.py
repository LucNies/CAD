# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:14:52 2016

@author: Luc
"""
from __future__ import division
from random import shuffle
import numpy as np
import patcher
import math


def get_features_labels(images, truth=None, patching=patcher.ImPatch()):
    features = []
    labels = []
    
    #alleen voor batchsize 1!!
    patches, coords = patching.patch(images)
    
    patches_border_dist = dist_to_border(patches, coords, images[0].shape)
    
    patches_center_dist = dist_to_center(patches_border_dist, coords, images[0].shape)
    
    features = patches_center_dist#aanpassen als we vaker willen
    
    shuffled = np.arange(len(features))
    shuffle(shuffled)
    features = reorder(features, shuffled)

    if truth is not None:
        for (x,y) in coords:
            labels.append(truth[0][x][y])
        labels = reorder(labels, shuffled)
        return features, labels
    
    return features

def calc_dice(predictions, labels):
    NNEQ = np.sum(predictions != labels)
    NNT = np.sum(predictions == labels)
    NNZ = np.sum((predictions + labels) != 0)
    return NNEQ/float(NNT+NNZ)

    
def reorder(array, order):
    result = [array[i] for i in order]
    return result
            
            
def dist_to_border(patches, coords, image_shape):
    result = []
    x_max = image_shape[0]
    y_max = image_shape[1]
    
    for i,patch in enumerate(patches):
        x = patch.append(coords[i][0])
        y = patch.append(coords[i][1])
        
        feature = min(x, x_max-x, y, y_max-y)
        patch_with_feature = patch.append(feature)
        result.append(patch_with_feature)
        
    return np.asarray(result)
    
    
def dist_to_center(patches, coords, image_shape):
    result = []
    x_center = image_shape[0]/2
    y_center = image_shape[1]/2
    
    for i,patch in enumerate(patches):
        x = patch.append(coords[i][0])
        y = patch.append(coords[i][1])
        
        feature = math.hypot(x-x_center, y-y_center)
        patch_with_feature = patch.append(feature)
        result.append(patch_with_feature)
        
    return np.asarray(result)    
































