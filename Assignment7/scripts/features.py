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
from scipy import misc
import pickle
from sklearn.externals import joblib
from sklearn.neighbors import DistanceMetric


loader = load_data.loader(first_run = False)


"""
Only pass one image at a time
"""
def get_patch_features(images, truth, patching):
    features = []
    labels = []
    

    patches, coords = patching.patch(images)
    features = patches
    for (x,y) in coords:
        labels.append(truth[x][y])
        
    
    shuffled = np.arange(len(labels))
    shuffle(shuffled)
    features = reorder(features, shuffled)
    labels = reorder(labels, shuffled)    
    
    return features, labels

def remove_black(features, labels):


    print "none"

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

def save_result_as_image(predictions, labels, file_path = '../images/'):
    for i, label in enumerate(labels):
        misc.imsave(str(i) + 'lable.png', label)
        misc.imsave(str(i) + 'prediction.png', predictions[i])
    
    
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