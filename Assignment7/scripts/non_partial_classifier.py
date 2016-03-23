# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:05:57 2016

@author: Luc
"""
from __future__ import division
import patcher
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pickle
from sklearn.externals import joblib
from sklearn.neighbors import DistanceMetric
import load_data2 as load_data
from features import get_features_labels
from features import calc_dice
import time
from plot_images import plot
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
class CLF:

    def __init__(self):
        self.loader = load_data.loader(first_run = True)
        self.patching = patcher.ImPatch()

    def train(self, clf =RandomForestClassifier()):
        print "Training SVC classifier"
        print "Part done:"
        loader = self.loader
        patching = self.patching
        self.clf = clf
        t1 = time.time()
        features, labels = loader.get_all_training_samples()
        clf.fit(features, labels)
        joblib.dump(self.clf, 'classifier.pkl')
        print "Training done, time elapsed: " + str(time.time()-t1)

    def test(self):
        #print "Testing SGD classifier"
        loader = self.loader
        self.clf = joblib.load('classifier.pkl')
        patching = self.patching
        accuracy = 0
        probabilities = []
        labels = []
 
        print "Start Testing..."
        print "Test size = " + str(loader.test_size)
        while loader.test_i < loader.test_size:
            feature_vector, label= loader.get_next_test_sample()
            prediction = self.clf.predict_proba(feature_vector)
            probabilities.append(prediction)
            labels.append(label)

        accuracy = np.zeros((10,))
        labels = np.array([l for ls in labels for l in ls])
        probabilities = np.array([p[1] for ps in probabilities for p in ps])

        for i,t in enumerate(np.arange(0,1,0.1)):
            # p[1] grabs P(class=1) the rest flattens and makes np-array for calc_dice
            predictions = probabilities >= t
            accuracy[i] = calc_dice(predictions, labels)
            print "t={}, dice: {}".format(t, str(accuracy[i]))
            print "Accurracy: "+str((predictions == labels).sum()/labels.size)
        return np.argmin(accuracy), np.min(accuracy)

    def classify(self, image):
        """ Gives confidence score for given image
        :param image: np vector containing an image
        :return: Confidence score between 0 and 1
        """
        features, labels = get_features_labels(image, patching=self.patching)
        prediction = self.clf.decision_function(features)
        return prediction
    

        

if __name__ == "__main__":
    sgd = CLF()
    sgd.train()
    #sgd.test()
