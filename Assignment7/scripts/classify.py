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

class CLF:

    def __init__(self):
        self.loader = load_data.loader(first_run = False)
        self.patching = patcher.ImPatch()

    def train(self, clf = linear_model.SGDClassifier()):
        print "Training SGD classifier"
        print "Part done:"
        loader = self.loader
        patching = self.patching
        self.clf = clf
        t1 = time.time()
        while loader.train_i < loader.train_size:# not testloader.reset:
            features, labels = loader.get_next_training_sample()
            self.clf.partial_fit(features, labels, [0,1])
            print loader.train_i/loader.train_size

        joblib.dump(self.clf, 'classifier.pkl')
        print "Training done, time elapsed: " + str(time.time()-t1)

    def test(self):
        #print "Testing SGD classifier"
        loader = self.loader
        self.clf = joblib.load('classifier.pkl')
        patching = self.patching
        accuracy = 0
        predictions = []
        labels = []
 
        print "Start Testing..."
        print "Test size = " + str(loader.test_size)
        while loader.test_i < loader.test_size:# not testloader.reset:
            feature_vector, label= loader.get_next_test_sample()
            prediction = self.clf.predict(feature_vector)#self.clf.decision_function(feature_vector)
            predictions.append(prediction)
            labels.append(label)
            print "Accuracy: " + str((prediction == label).sum()/label.size)
            print "True positives:" +str((prediction+label == 2).sum()/label.sum())
            print str(loader.test_i/loader.test_size)
            
            #features = np.reshape(features, (-1, np.shape(features)[-1]))
            #print calc_dice(prediction, labels)
        
        """
        accuracy = np.zeros((10,))
        for i,t in enumerate(np.arange(0,1,0.1)):
            predictions = predictions >= t
            accuracy[i] = calc_dice(predictions, labels)
            print "Accurracy: "+str((predictions == labels).sum()/labels.size)
            print "t={}, Mean error(dice): ".format(t) + str(accuracy[i])
        return np.argmin(accuracy), np.min(accuracy)
        """

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
    #sgd.train(clf = linear_model.SGDClassifier())
    sgd.test()
