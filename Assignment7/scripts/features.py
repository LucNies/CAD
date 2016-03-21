# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:14:52 2016

@author: Luc
"""
from __future__ import division
from random import shuffle
from sklearn import linear_model
from scipy import misc
import pickle
from sklearn.externals import joblib
from sklearn.neighbors import DistanceMetric


import numpy as np
import patcher
import math


def get_features_labels(images, truth=None, patching=patcher.ImPatch()):

    features = []
    labels = []
    

    patches, coords = patching.patch(images)

    
    patches_border_dist = dist_to_border(patches, coords, images[0].shape)
    
    patches_center_dist = dist_to_center(patches_border_dist, coords, images[0].shape)
    
    features = patches_center_dist#aanpassen als we vaker willen
    
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
        labels = [l!=0 for l in labels]
        return features, labels
    
    return features


def train():
    clf = linear_model.SGDClassifier()
    patching = patcher.ImPatch()
    train_images, train_labels = loader.get_train_data()
    for i, image in enumerate(train_images):
        features, labels = get_patch_features(image, train_labels[i], patching)
        clf.partial_fit(features, labels, [0,1])
        print i/len(train_images)
    
    joblib.dump(clf, 'classifier.pkl')
    
def test():
    clf = joblib.load('classifier.pkl')  
    patching = patcher.ImPatch()
    test_images, test_labels = loader.get_test_data()
    predictions = np.zeros(test_labels.shape)
    for i, image in enumerate(test_images):
        features, labels = get_patch_features(image, test_labels[i], patching)
        prediction = clf.predict(features)
        #predictions[i] = prediction werkt niet, want randenm
        print calc_dice(prediction, labels)
        
    #save_result_as_image(predictions, labels)


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
            
            
def dist_to_border(patches, coords, image_shape):
    x_max = image_shape[0]
    y_max = image_shape[1]
    result = np.zeros((patches.shape[0],patches.shape[1]+1))

    for i,patch in enumerate(patches):
        x = coords[i][0]
        y = coords[i][1]
        
        feature = min(coords[i][0], x_max-x, y, y_max-y)
        result[i,:patches.shape[1]] = patch
        result[i, patches.shape[1]:] = feature

    return result
    
    
def dist_to_center(patches, coords, image_shape):
    x_center = image_shape[0]/2
    y_center = image_shape[1]/2
    result = np.zeros((patches.shape[0],patches.shape[1]+1))
    
    for i,patch in enumerate(patches):
        x = coords[i][0]
        y = coords[i][1]
        
        feature = math.hypot(x-x_center, y-y_center)
        result[i,:patches.shape[1]] = patch
        result[i, patches.shape[1]:] = feature
        
    return result




