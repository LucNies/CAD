# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:05:57 2016

@author: Luc
"""

import patcher
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pickle
from sklearn.externals import joblib
from sklearn.neighbors import DistanceMetric
import load_data
from features import get_features_labels, calc_dice

class SGD:

    def __init__(self):
        self.loader = load_data.loader(batch_size = 1)
        self.clf = None
        self.patching = patcher.ImPatch()

    def train(self):
        print "Training SGD classifier"
        print "Part done:"
        loader = self.loader
        clf = linear_model.SGDClassifier()
        patching = self.patching
        while loader.batch_i < loader.n_batch/20:
            data, truth = loader.load_batch()
            features, labels = get_features_labels(data, truth, patching)
            clf.partial_fit(features, labels, [0,1])
            print loader.batch_i/float(loader.n_batch)

        joblib.dump(clf, 'classifier.pkl')
        self.clf = clf

    def test(self, t=0.5):
        #print "Testing SGD classifier"
        loader = self.loader
        if self.clf is None:
            clf = joblib.load('classifier.pkl')
        else:
            clf = self.clf
        patching = self.patching
        testloader = load_data.loader(batch_size = 1)
        accuracy = []
        while testloader.batch_i < loader.n_batch:
            data, truth = loader.load_batch()
            features = []
            for d,t in zip(data, truth):
                feature, labels = get_features_labels(d, t, patching)
                features.append(feature)
            prediction = clf.decision_function(features)
            prediction = prediction >= t

            #print calc_dice(prediction, labels)
            accuracy.append(calc_dice(prediction, labels))
        print "Mean accuracy(dice): " + str(np.mean(accuracy))
        return np.mean(accuracy)

    def classify(self, image):
        """ Gives confidence score for given image
        :param image: np vector containing an image
        :return: Confidence score between 0 and 1
        """
        features, labels = get_features_labels(image, patching=self.patching)
        prediction = self.clf.decision_function(features)
        return prediction

if __name__ == "__main__":
    sgd = SGD()
    #sgd.train()
    sgd.test()
