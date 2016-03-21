# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:00:14 2016

@author: Luc
"""


import numpy as np
import os
from scipy import misc





class loader():

    def __init__(self, file_path='../data/', batch_size = 20):
        self.file_path = file_path
        file_names = os.listdir(file_path)
        file_names = [nm for nm in os.listdir(file_path) if nm[-4:]==".png"]
        self.truth_names = np.array([nm for nm in file_names if nm[-6:]=="an.png"])
        file_names = np.array([nm for nm in file_names if nm[-6:]!="an.png"])
        self.file_names = np.array(file_names).reshape((len(file_names)/3,3))
        #self.truth_names = self.file_names[:,1]
        #self.file_names = self.file_names[:,1:]
        self.batch_size = batch_size
        self.batch_i = 0
        self.n_batch = len(self.truth_names)/batch_size
        image = misc.imread(os.path.join(file_path,self.file_names[0][0]))
        self.im_shape = image.shape
        self.reset = False


    def load_batch(self):
        self.reset = False
        if (self.batch_i+1)*self.batch_size > len(self.truth_names):
            batch_size = len(self.truth_names) - self.batch_i*self.batch_size
            self.reset = True
        else:
            batch_size = self.batch_size
            self.reset = False
            
        data = np.zeros((3*batch_size,) + self.im_shape)
        truth = np.zeros((batch_size,) + self.im_shape)
        for i in range(batch_size):
            data[3*i  ] = misc.imread(self.file_path+self.file_names [self.batch_i*batch_size + i,0])
            data[3*i+1] = misc.imread(self.file_path+self.file_names [self.batch_i*batch_size + i,1])
            data[3*i+2] = misc.imread(self.file_path+self.file_names [self.batch_i*batch_size + i,2])
            truth[i]    = misc.imread(self.file_path+self.truth_names[self.batch_i*batch_size + i])
        self.batch_i = 0 if self.reset else self.batch_i+1
        return data, truth
        


if __name__ == "__main__":
    loader = loader(first_run=False)
    train_images, train_labels = loader.get_train_data()
    test_images, test_labels = loader.get_test_data()
    print "done"
