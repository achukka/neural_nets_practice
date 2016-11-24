# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

''' A Library to load the mnist image '''
import cPickle
import gzip

# <codecell>

import numpy as np

# <codecell>

''' A simple normal load function
    Each image contains 28 * 28 pixels = 784 values. 
    Hence we would represent it as a vector of 784 values.
    training_data - numpy array of 50,000 images
    validation_data, test_data - numpy array of 10,000 images.
    Note that both validation_data and test_data are the same in this case.
'''
def load_data():
    fp = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(fp)
    fp.close()
    return (training_data, validation_data, test_data)

# <codecell>

''' The above input representation might not work for neural networks. 
    Hence we will modify it a little using the below function
    A Wrapper for the normal load function
'''
def load_data_wrapper():
    tr_data, vl_data, tst_data = load_data()
    training_data_inputs = [np.reshape(x, (784, 1)) for x in tr_data[0]]
    training_data_results = [vectorized(y) for y in tr_data[1]]
    training_data = zip(training_data_inputs, training_data_results)
    validation_inputs = [ np.reshape(x, (784, 1)) for x in vl_data[0]]
    validation_data = zip(validation_inputs, vl_data[1])
    test_inputs = [ np.reshape(x, (784, 1)) for x in tst_data[0]]
    test_data = zip(test_inputs, tst_data[1])
    return (training_data, validation_data, test_data)

''' Returns a 10 dimensional vector with dth dimnesion set to 1.0 and rest are zero
'''
def vectorized(d):
    vec = np.zeros((10, 1))
    vec[d] = 1.0
    return vec

# <codecell>

training_data, validation_data, test_data = load_data_wrapper()

# <codecell>

len(training_data), len(test_data), len(validation_data)

# <codecell>


