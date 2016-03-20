# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:14:52 2016

@author: Luc
"""
from __future__ import division
import load_data
import patcher
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from sklearn import linear_model
import pickle
from sklearn.externals import joblib
from sklearn.neighbors import DistanceMetric

loader = load_data.loader(batch_size = 1)



def get_features_labels(images, truth, patching):
    features = []
    labels = []
    
    #alleen voor batchsize 1!!
    patches, coords = patching.patch(images)
    features = patches#aanpassen als we vaker willen
    for (x,y) in coords:
        labels.append(truth[0][x][y])
        
    
    shuffled = np.arange(len(labels))
    shuffle(shuffled)
    features = reorder(features, shuffled)
    labels = reorder(labels, shuffled)    
    
    return features, labels


def train():
    clf = linear_model.SGDClassifier()
    patching = patcher.ImPatch()
    while loader.batch_i < loader.n_batch:
        data, truth = loader.load_batch()
        features, labels = get_features_labels(data, truth, patching)
        clf.partial_fit(features, labels, [0,1])
        print loader.batch_i/loader.n_batch
    
    joblib.dump(clf, 'classifier.pkl')
    
def test():
    clf = joblib.load('classifier.pkl')  
    patching = patcher.ImPatch()
    testloader = load_data.loader(batch_size = 1)
    while testloader.batch_i < loader.n_batch:
        data, truth = loader.load_batch()
        features, labels = get_features_labels(data, truth, patching)
        prediction = clf.predict(features)

        print calc_dice(prediction, labels)
        
    
def calc_dice(predictions, labels):
    NNEQ = np.sum(predictions != labels)
    NNT = np.sum(predictions == labels)
    NNZ = np.sum((predictions + labels) != 0)
    return NNEQ/(NNT+NNZ)
        
    

    
def reorder(array, order):
    result = [array[i] for i in order]
    return result
            
def create_location_feature():
        loader = load_data.loader()
        data, truth = loader.load_batch()
        locations = np.zeros()
        
#train()
test()