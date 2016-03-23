# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:14:52 2016

@author: Luc
"""
from __future__ import division
from random import shuffle
from sklearn import linear_model
from scipy import misc
from sklearn.externals import joblib
from sklearn.neighbors import DistanceMetric

import load_data2
import numpy as np
import patcher
import math


def get_features_labels(image, truth=None, patching=patcher.ImPatch()):

    features = []
    labels = []
    

    patches, coords = patching.patch(image)
    
    
    #patches_border_dist = dist_to_border(patches, coords, image.shape)
    
    #patches_center_dist = dist_to_center(patches_border_dist, coords, image.shape)
    
    #features = patches_center_dist#aanpassen als we vaker willen
    

    patches, coords = remove_threshold(patches, coords)  

    for (x,y) in coords:
        labels.append(truth[x][y]>240)#Not really a binary image: contains values {0,1,244,255}
    

    
    shuffled = np.arange(len(patches))
    shuffle(shuffled)
    features = reorder(patches, shuffled)
    labels = reorder(labels, shuffled)
     
    
     
    
    return features, labels
    
"""
Removes pixels that are below the threshold from the feature vector
"""
def remove_threshold(features, coords, t = 1):
    to_remove = []
    for i, feature in enumerate(features):
         if feature[len(feature)/3/2]<=0 and feature[2*(len(feature)/3/2)]<=0 and feature[3*(len(feature)/3/2)]<=0 :
             to_remove.append(i)
            
    np.delete(features, to_remove)
    np.delete(coords, to_remove)
    return features, coords
             
    
"""
Do not use
def train():
    clf = linear_model.SGDClassifier()
    patching = patcher.ImPatch()
    train_images, train_labels = loader.get_train_data()
    for i, image in enumerate(train_images):
        features, labels = get_patch_features(image, train_labels[i], patching)
        clf.partial_fit(features, labels, [0,1])
        print i/len(train_images)
    
    joblib.dump(clf, 'classifier.pkl')
 

Do not use
   
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

"""


def calc_dice(predictions, labelsi, similarity=True):
    NTT = np.sum(predictions * labels)
    if similarity:
        N = np.sum(predictions) + np.sum(labels)
        return 2*NTT/N
    else:
        NNEQ = np.sum(predictions != labels)
        NNZ = np.sum((predictions + labels) != 0)
        return NNEQ/float(NTT+NNZ) # Dissimilarity

            
    
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


if __name__ == "__main__"  :
    loader = load_data2.loader(first_run = True)
    images, labels = loader.get_test_data()
    get_features_labels(images[0], labels[0])
