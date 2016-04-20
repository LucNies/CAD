# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:30:16 2016

@author: luc
"""

import numpy as np
import gzip
import matplotlib.pyplot as plt
#%matplotlib inline
import cPickle as pickle
import time
import theano
import theano.tensor as T
import lasagne
from math import sqrt, ceil
import os

dataset_dir = "../data/"

class learn_cifar:
    
    def __init__(self, dataset_dir = "../data/", filter_size = (5,5), num_filters = 32, dense_units=128, weights = lasagne.init.GlorotUniform(), n_epochs = 20, learning_rate = 0.01, test=False, n_batches = 4):
        self.dataset_dir = dataset_dir
        self.prepare_trainings_data(n_batches = n_batches)
        inputs, targets, network = self.create_network(filter_size = (5,5), num_filters = 32, dense_units=128, weights = lasagne.init.GlorotUniform())
        self.curves, test_fn = self.training(inputs, targets, network, n_epochs = n_epochs, learning_rate = learning_rate)
        if(test):            
            self.run_test_set(test_fn)

    def prepare_trainings_data(self, n_batches = 4):
        # training set, batches 1-4
        n_samples = 10000 # number of samples per batch
        self.train_X = np.zeros((n_samples*n_batches, 3, 32, 32), dtype="float32")
        self.train_Y = np.zeros((n_samples*n_batches, 1), dtype="ubyte").flatten()
    
        for i in range(n_batches):
            f = open(os.path.join(self.dataset_dir, "data_batch_"+str(i+1)+""), "rb")
            cifar_batch = pickle.load(f)
            f.close()
            self.train_X[i*n_samples:(i+1)*n_samples] = (cifar_batch['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
            self.train_Y[i*n_samples:(i+1)*n_samples] = np.array(cifar_batch['labels'], dtype='ubyte')
            
        # validation set, batch 5    c
        f = open(os.path.join(self.dataset_dir, "data_batch_5"), "rb")
        cifar_batch_5 = pickle.load(f)
        f.close()
        self.val_X = (cifar_batch_5['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
        self.val_Y = np.array(cifar_batch_5['labels'], dtype='ubyte')
        
        # labels
        f = open(os.path.join(dataset_dir, "batches.meta"), "rb")
        cifar_dict = pickle.load(f)
        label_to_names = {k:v for k, v in zip(range(10), cifar_dict['label_names'])}
        f.close()
        
        self.normalize_dataset()
        print("training set size: data = {}, labels = {}".format(self.train_X.shape, self.train_Y.shape))
        print("validation set size: data = {}, labels = {}".format(self.val_X.shape, self.val_Y.shape))
    
    def normalize_dataset(self):
        train_mean = np.mean(self.train_X)
        train_std = np.std(self.train_X)
        self.train_X = (self.train_X-train_mean)/train_std
        self.val_X = (self.val_X-train_mean)/train_std#means and std are proabaly the same in the train and validationset
        
    # take an array of shape (n, height, width) or (n, height, width, channels)

    def run_test_set(self, test_fn):
        with open(os.path.join(self.dataset_dir, "test_batch"), "rb") as f:
            cifar_test_batch = pickle.load(f)
        
        inputs = T.tensor4('X')
        targets = T.ivector('y')        
        
        test_X = (cifar_test_batch['data'].reshape(-1, 3, 32, 32) / 255.).astype("float32")
        test_Y = np.array(cifar_test_batch['labels'], dtype='ubyte')
        test_mean = np.mean(test_X)
        test_std = np.std(test_X)
        test_X = (test_X-test_mean)/test_std
                
        accuracy  = 0
        for i, batch in enumerate(self.iterate_minibatches(test_X, test_Y, 500, shuffle=False)):
           inputs, targets = batch
           preds, err, acc = test_fn(inputs, targets)
           accuracy += acc
               
        print "test set accuracy: " + str(accuracy/i)  +"%"
        
        

# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """
    Copied without testing
    """
    def visualize_data(self, data, padsize=1, padval=0, cmap="gray", image_size=(10,10)):
    
        data -= data.min()
    
        data /= data.max()
    
        
    
        # force the number of filters to be square
    
        n = int(np.ceil(np.sqrt(data.shape[0])))
    
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
        
    
        # tile the filters into an image
    
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
        
    
        plt.figure(figsize=image_size)
    
        plt.imshow(data, cmap=cmap)
    
        plt.axis('off')


    def create_network(self, filter_size = (4,4), learning_rate = 0.01, num_filters = 8, dense_units=128, weights = lasagne.init.GlorotUniform()):
        # First we define the symbolic input X and the symbolic target y. We want
        # to solve the equation y = C(X) where C is a classifier (convolutional network).
        inputs = T.tensor4('X')
        targets = T.ivector('y')

        # Input layer
        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=inputs)

        print lasagne.layers.get_output_shape(network)
        # Convolutional layer
        network = lasagne.layers.Conv2DLayer(network, num_filters=num_filters, filter_size=filter_size, nonlinearity=lasagne.nonlinearities.rectify, W=weights)
        print lasagne.layers.get_output_shape(network)

        # Max-pooling layer
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2)
        print lasagne.layers.get_output_shape(network)

        # Convolutional layer
        network = lasagne.layers.Conv2DLayer(
                    network, num_filters=num_filters*2, filter_size=filter_size,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=weights)
        print lasagne.layers.get_output_shape(network)

        # Max-pooling layer
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))
        print lasagne.layers.get_output_shape(network)

        # Convolutional layer
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=num_filters*4, filter_size=filter_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=weights)
        print lasagne.layers.get_output_shape(network)

        # Max-pooling layer
        #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))
        print lasagne.layers.get_output_shape(network)

        # Fully-connected (dense) layer
        network = lasagne.layers.DenseLayer(
                    network,
                    num_units=dense_units,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.Orthogonal())
        print lasagne.layers.get_output_shape(network)

        # Soft-max layer
        network = lasagne.layers.DenseLayer(
                    network, num_units=10,
                    nonlinearity=lasagne.nonlinearities.softmax)
        print lasagne.layers.get_output_shape(network)

        return inputs, targets, network

# ### Batch iterator ###
# This is just a simple helper function iterating over training 
# data in mini-batches of a particular size, optionally in random order. 
# It assumes data is available as numpy arrays.
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=True):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
                #data augmentation
                for idx in excerpt:
                    if np.random.randint(2) > 0:                    
                        inputs[idx]=np.fliplr(inputs[idx])
            else:
                excerpt = slice(start_idx, start_idx + batchsize)      
            
            yield inputs[excerpt], targets[excerpt]


    def training(self, inputs, targets, network, n_epochs = 20, learning_rate = 0.01):
        
        # First we get the prediction from the last layer and then calculate the
        # the loss for each sample and take the mean as final loss.
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, targets)
        loss = loss.mean()        
        
        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD), but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.sgd(loss, params, learning_rate=learning_rate)


        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, targets)
        test_loss = test_loss.mean()
        acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), targets),
                    dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([inputs, targets], loss, updates=updates)
        
        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([inputs, targets], [test_prediction, test_loss, acc])

        begin = time.time()
        print "Start training" 
        # The number of epochs specifies the number of passes over the whole training data
        curves = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        for epoch in range(n_epochs):
            # In each epoch, we do a full pass over the training data...
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(self.train_X, self.train_Y, 32, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1
        
            # ...and a full pass over the validation data
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(self.val_X, self.val_Y, 500, shuffle=False):
                inputs, targets = batch
                preds, err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
        
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
            curves['train_loss'].append(train_err / train_batches)
            curves['val_loss'].append(val_err / val_batches)
            curves['val_acc'].append(val_acc / val_batches)
    
        print "Total runtime: " +str(time.time()-begin)
        
        return curves, val_fn 

    
        
    
    def save_result(self, file_path = '../plots/', name = ''):
        curves = self.curves
        plt.plot(zip(curves['train_loss'], curves['val_loss']));
        plt.savefig(file_path + name + 'loss.png')
        plt.clf()
        plt.plot(curves['val_acc']);
        plt.savefig(file_path + name + 'accuracy.png')
        plt.clf()
        print "saved plots" 
        
if __name__ == '__main__':
    #find learning rate
    #network = learn_cifar(n_batches = 1 , n_epochs = 3, test=True)
    #network.save_result(name = 'sgd-learning_rate{}'.format(i))

    alphas = [0.001, 0.05, 0.01, 0.05, 0.1]
    for i in alphas:
        network = learn_cifar(learning_rate = i)
        network.save_result(name = 'sgd-learning_rate{}'.format(i))

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        