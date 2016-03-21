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

class CLF:

    def __init__(self):
        self.loader = load_data.loader(batch_size = 1, clf = linear_model.SGDClassifier())
        self.clf = clf
        self.patching = patcher.ImPatch()

    def train(self):
        print "Training SGD classifier"
        print "Part done:"
        loader = self.loader
        clf = self.clf
        patching = self.patching
        while not loader.reset:
            data, truth = loader.load_batch()
            features, labels = get_features_labels(data, truth, patching)
            clf.partial_fit(features, labels, [0,1])
            print loader.batch_i/float(loader.n_batch)

        joblib.dump(clf, 'classifier.pkl')

    def test(self, t=0.5):
        #print "Testing SGD classifier"
        loader = self.loader
        if self.clf is None:
            clf = joblib.load('classifier.pkl')
        else:
            clf = self.clf
        patching = self.patching
        testloader = load_data.loader(batch_size = 1)
        accuracy = 0
        predictions = np.zeros((testloader.n_batch, testloader.batch_size * patching.nmaxpatches))
        labels = np.zeros((testloader.n_batch, testloader.batch_size * patching.nmaxpatches))
        i = 0
        while loader.batch_i <= 1:# not testloader.reset:
            data, truth = loader.load_batch()
            feature, label = get_features_labels(data, truth, patching)
            prediction = clf.decision_function(feature)
            predictions[i] = prediction
            labels[i] = label
            i+=1
            #features = np.reshape(features, (-1, np.shape(features)[-1]))
            #print calc_dice(prediction, labels)
        accuracy = np.zeros((10,))
        for i,t in enumerate(np.arange(0,1,0.1)):
            predictions = predictions >= t
            accuracy[i] = calc_dice(predictions, labels)
            print "t={}, Mean error(dice): ".format(t) + str(accuracy[i])
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
    sgd = CLF(clf = linear_model.SGDClassifier())
    #sgd.train()
    sgd.test()
