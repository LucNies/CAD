# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:00:14 2016

@author: Luc
"""

from __future__ import division
import numpy as np
import os
from scipy import misc

import features
from random import shuffle




class loader():

    def __init__(self, file_path='../data/', test_size = 10, first_run = True):
        self.train_i = 0
        self.test_i = 0
        if first_run:
            self.test_size = test_size
            self.train_size = 50 - self.test_size
            self.first_time(file_path)
        
        else:
            shape = np.load('test_labels.npy').shape
            self.test_size = shape[0]
            self.im_shape = shape[1:]
            self.train_size = 50-self.test_size

        


    def first_time(self, file_path):
        print "loading data first time"
        file_names = os.listdir(file_path)  
        file_names = np.array(file_names).reshape((len(file_names)/4,4))
        image = misc.imread(file_path+file_names[0][0])
        self.im_shape = image.shape

        #split test and train set
        np.random.shuffle(file_names)
        truth_names = file_names[:,0]
        truth_names_test = truth_names[:10]
        truth_names_train = truth_names[10:]

        file_names = file_names[:,1:]
        file_names_test = file_names[:10]
        file_names_train = file_names[10:]
        
        np.save('test_labels.npy', self.load_labels(file_path, truth_names_test))
        np.save('train_labels.npy', self.load_labels(file_path, truth_names_train))
        np.save('test_images.npy', self.load_images(file_path, file_names_test))
        np.save('train_images.npy', self.load_images(file_path, file_names_train))
        
        print "saved images as numpy arrays"
        
        self.save_features()
        print "Saved features and labels to file"

    def load_labels(self, path, file_names):
        data = np.zeros((len(file_names),) + self.im_shape)
        for i, name in enumerate(file_names):
            data[i] = misc.imread(path+name);
        
        return data
    
    def load_images(self, path, file_names):
        data = np.zeros((len(file_names), 3) + self.im_shape)
        for i in range(len(file_names)):
            data[i][0] = misc.imread(path+file_names[i][0]);
            data[i][1] = misc.imread(path+file_names[i][1]);
            data[i][2] = misc.imread(path+file_names[i][2]);
        
        return data
      
    def get_train_data(self):
        images = np.load('train_images.npy')
        labels = np.load('train_labels.npy')
        print "loaded train data"
        return images, labels
    
    def get_test_data(self):
        images = np.load('test_images.npy')
        labels = np.load('test_labels.npy')
        print "loader test data"
        return images, labels

    def save_features(self, file_path = '../features/'):
        
        train_images, train_labels = self.get_train_data()
        test_images, test_labels = self.get_test_data()

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        print "Patching and saving train images"
        for i, image in enumerate(train_images):
            label = train_labels[i]
            feature_vectors, labels = features.get_features_labels(image, label)
            feature_vectors, labels = self.subsample(feature_vectors, labels)
            np.save(file_path + "train_features_n" + str(i) + '.npy', feature_vectors)
            np.save(file_path + "train_labels_n" + str(i) + '.npy', labels)
            print i/len(train_images)
        
        print "Patching and saving test images"
        for i, image in enumerate(test_images):
            label = test_labels[i]
            feature_vectors, labels = features.get_features_labels(image, label)
            feature_vectors, labels = self.subsample(feature_vectors, labels)
            np.save(file_path + "test_features_n" + str(i) + '.npy', feature_vectors)
            np.save(file_path + "test_labels_n" + str(i) + '.npy', labels)
            print i/len(test_images)
        
    def get_next_training_sample(self, file_path = '../features/'):
         feature_vectors = np.load(file_path + 'train_features_n' + str(self.train_i) + '.npy')
         labels = np.load(file_path + 'train_labels_n' + str(self.train_i) + '.npy')
         self.train_i =self.train_i+1
         return feature_vectors, labels
        
    def get_next_test_sample(self, file_path = '../features/'):
         feature_vectors = np.load(file_path + 'test_features_n' + str(self.test_i) + '.npy')
         labels = np.load(file_path + 'test_labels_n' + str(self.test_i) + '.npy')
         self.test_i = self.test_i+1
         return feature_vectors, labels

    def subsample(self, objects, labels, ratio=1):
        """Samples objects and corresponding labels such that sum(labels==0)=sum(labels==1)
        :param objects: Objects to sample.
        :param labels: Binary labels to weigh the sampling, are also sampled.
        :return: Subsample of objects and corresponding labels
        """

        objects = np.array(objects)
        labels = np.array(labels)

        n1 = np.sum(labels)
        n0 = np.size(labels,0) - n1
        r = n0/n1

        ind0 = np.where(1-labels)[0]
        ind1 = np.nonzero(labels)[0]
        if r > 0:
            ind0 = np.random.choice(ind0, n1)
        else:
            ind1 = np.random.choice(ind1, n0)
        oobj = np.vstack([objects[ind0,:], objects[ind1,:]])
        olbl = np.append(labels[ind0], labels[ind1])
        shuffled = np.arange(len(olbl))
        shuffle(shuffled)
        oobj = oobj[shuffled]
        olbl = olbl[shuffled]

        return oobj, olbl


if __name__ == "__main__":
    loader = loader(first_run=False)
    #train_images, train_labels = loader.get_train_data()
    #test_images, test_labels = loader.get_test_data()
    print "done"
