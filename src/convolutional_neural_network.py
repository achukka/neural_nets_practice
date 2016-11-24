
# coding: utf-8

# In[1]:

# Standard Libraries
import cPickle
import gzip

# External Libraries
import numpy as np
import theano
import theano.tensor as Tensor
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample


# In[2]:

# Define Activation functions
# Linear Neuron
def linear(z):
    return z

# Rectified Linear Unit
def ReLu(z):
    return Tensor.maximum(0.0, z)

# Tanh Neuron
def tanh(z):
    return Tensor.tanh(z)

# Sigmoid Function
def sigmoid(z):
    return 1.0/(1+np.exp(-z))


# In[3]:

GPU = True
if GPU:
    print 'Trying to run under a GPU. If this is not required, '+            'Set GPU flag to False'
    theano.config.mode = 'FAST_RUN'
    try:
        theano.config.device = 'gpu'
    except:
        pass # Already set
    theano.config.floatX = 'float32'
else:
    print 'Runinng under CPU. Inorder run on a GPU Set GPU flag to True'


# In[29]:

# Load MNIST Data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    fp = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(fp)
    fp.close()
    return [shared(training_data), shared(validation_data), shared(test_data)]
    
def shared(data):
    ''' Shared data for Theano library. It places the data onto a GPU
    if available'''
    shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX),
                             borrow=True)
    return shared_x, Tensor.cast(shared_y, "int32")
    


# In[22]:

# Main class for the network
# Neural Network
class Network(object):
    def __init__(self, layers, mini_batch_size):
        ''' Take a list of 'layers' to construct the network and the 
        value for 'mini_batch_size' to run stochastic gradient descent
        '''
        
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.parameters = [ parameter for layer in self.layers for parameter in layer.parameters]
        self.x = Tensor.matrix('x')
        self.y = Tensor.ivector('y')
        init_layer = self.layers[0]
        init_layer.set_input(self.x, self.x, self.mini_batch_size)
        for num in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[num-1], self.layers[num]
            layer.set_input(prev_layer.output, prev_layer.output_dropout,
                           self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        
    # Let us define the Stochastic Gradient Descent method.
    # The net uses SGD to learn the weights and biases
    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            validation_data, lmda=0.0):
        ''' 
            Trains the neural network using mini-batch stochastic gradient descent.
            The training data is list a tuples (x,y) representing the training input and desired outputs
            If validation data is provided the network will be evaluated against after each epoch,
        '''
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        
        # Compute the number of batches for training, validation, test
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        
        # Define the cost (regularized) function, gradients, updates
        l2_norm_squared = sum([(layer.weights**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) +            0.5*lmda*l2_norm_squared/num_training_batches
        gradients = Tensor.grad(cost, self.parameters)
        updates = [(parameter, parameter - eta*gradient)
                   for parameter, gradient in zip(self.parameters,gradients)]
        
        # Functions to train the mini batch and to compute the 
        # Accuracies in validation and test mini batches
        index = Tensor.lscalar() # mini-batch index
        train_mini_batch = theano.function(
            [index], cost, updates=updates,
            givens={
                self.x: training_x[index*self.mini_batch_size:(index+1)*self.mini_batch_size],
                self.y: training_y[index*self.mini_batch_size:(index+1)*self.mini_batch_size]
            })
        validation_mb_accuracy = theano.function(
            [index], self.layers[-1].accuracy(self.y),
            givens={
                self.x: validation_x[index*self.mini_batch_size:(index+1)*self.mini_batch_size],
                self.y: validation_y[index*self.mini_batch_size:(index+1)*self.mini_batch_size]
            })

        # Perform the actual training
        best_validation_accuracy = 0.0
        best_iteration = 0
        for epoch in xrange(epochs):
            for mini_batch_index in xrange(num_training_batches):
                iteration = num_training_batches * epoch + mini_batch_index
                cost_xy = train_mini_batch(mini_batch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validation_mb_accuracy(num) for num in xrange(num_validation_batches)])
                    if validation_accuracy >= best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
        print "Finished Training Network"
        print 'Best Validation Accuracy of {0:.2%} acheived at iteration {1}'.format(
                best_validation_accuracy, best_iteration)
        return best_validation_accuracy

    def get_predictions(self, test_data):
        test_x, test_y = test_data
        index = Tensor.lscalar() # mini-batch index
        test_mb_predictions = theano.function(
            [index], self.layers[-1].y_out,
            givens={
                self.x : test_x[index]
            })
        return [np.argmax(test_mb_predictions(num)) for num in xrange(size(test_data))]
# In[15]:

class ConvPoolLayer(object):
    '''
    Creates a combination of Convolutional Layer and Max Pooling
    Note : A better implementation might treat them separately.
    '''
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=sigmoid):
        ''' 
        Initializes the layer with the following things
        'filter_shape' - a tuple (length-4) that consists of the number of filters,
                        number of feature maps, filter height, filter width
        'image_shape' - a tuple (length-4) that consists of the 'mini_batch_size',
                        number of feature maps, image height, image width
        'poolsize'    - a tuple (length-2) that consists of y and x pooling sizez
        '''
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.activation_fn = activation_fn
        self.poolsize = poolsize
        # Initialize weights and biases
        ''' n_out = num_features*height*width/(poolsize_x * poolsize_y)
            Ex: A Convolutional layer with 20 filters and 24 x 24 local receptive field
            with a 2 x2 max pool unit gives the output as 20 x 12 x 12
        '''
        n_out = (filter_shape[0]*np.prod(filter_shape[2:]))/np.prod(poolsize)
        self.weights = theano.shared(
                        np.asarray(
                            np.random.normal(
                                loc=0.0, scale=np.sqrt(1.0/n_out), size=(filter_shape)),
                            dtype=theano.config.floatX),
                        name='weights', borrow=True)
        self.biases = theano.shared(
                        np.asarray(
                            np.random.normal(
                                loc=0.0, scale=1.0, size=(filter_shape[0], )),
                            dtype=theano.config.floatX),
                        name='biases', borrow=True)
        self.parameters = [ self.weights, self.biases]
        
    def set_input(self, inpt, input_dropout, mini_batch_size):
        '''
        Sets the input for CONV Layer, by reshaping it image_shape format
        sets the output by forward pass using  the 'activation_fn'. 
        NO DROPOUT IN COVOLUTIONAL LAYER
        '''
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
                input=self.inpt, filters = self.weights, filter_shape=self.filter_shape,
                image_shape = self.image_shape)
        pooled_out = downsample.max_pool_2d(
                input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
                    pooled_out + self.biases.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # NO drop out in convolutional layers


# In[16]:

class FullyConnectedLayer(object):
    
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        ''' 
        Initializes the layer with 'n_in' inputs and 'n_out' output connections
        with the provided 'activation_fn' and 'p_dropout' rate
        '''
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.weights = theano.shared(
                        np.asarray(
                            np.random.normal(
                                loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                            dtype=theano.config.floatX),
                        name='weights', borrow=True)
        self.biases = theano.shared(
                        np.asarray(
                            np.random.normal(
                                loc=0.0, scale=1.0, size=(n_out, )),
                            dtype=theano.config.floatX),
                        name='biases', borrow=True)
        self.parameters = [ self.weights, self.biases]
        
    def set_input(self, inpt, input_dropout, mini_batch_size):
        '''
        Sets the input for the FC Layer, by reshaping it matrix of size
        'mini_batch_size' x 'n_in', sets the output by forward pass using 
        the 'activation_fn'. 'input_dropout' and 'output_dropout' are set using 
        the dropout layer prescribed earlier.
        '''
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
                    ( 1 - self.p_dropout)* Tensor.dot(self.inpt, self.weights) + self.biases)
        self.y_out = Tensor.argmax(self.output, axis=1)
        self.input_dropout = dropout_layer(
                input_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
                Tensor.dot(self.input_dropout, self.weights) + self.biases)
        
    def accuracy(self, y):
        '''
        Returns the accuracy for the mini-batch
        '''
        return Tensor.mean(Tensor.eq(y , self.y_out))


# In[17]:

class SoftMaxLayer(object):
    
    def __init__(self, n_in, n_out, p_dropout=0.0):
        ''' 
        Initializes the layer with 'n_in' inputs, 'n_out' output connections
        ad 'p_dropout' rate
        '''
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.weights = theano.shared(
                        np.zeros((n_in, n_out), dtype=theano.config.floatX),
                        name='weights', borrow=True)
        self.biases = theano.shared(
                        np.zeros((n_out,), dtype=theano.config.floatX),
                        name='biases', borrow=True)
        self.parameters = [ self.weights, self.biases]
        
    def set_input(self, inpt, input_dropout, mini_batch_size):
        '''
        Sets the input for the Spftmax Layer Layer, by reshaping it matrix of size
        'mini_batch_size' x 'n_in', sets the output by forward pass using 
        the 'activation_fn'. 'input_dropout' and 'output_dropout' are set using 
        the dropout layer prescribed earlier.
        '''
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1 - self.p_dropout)* Tensor.dot(self.inpt, self.weights)
                              + self.biases)
        self.y_out = Tensor.argmax(self.output, axis=1)
        self.input_dropout = dropout_layer(
                input_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(Tensor.dot(self.input_dropout, self.weights)
                                      + self.biases)
    
    def cost(self, net):
        '''
        Returns the log-likelihood cost.
        '''
        return -Tensor.mean(Tensor.log(self.output_dropout)[Tensor.arange(
                                                    net.y.shape[0]), net.y])
    
    def accuracy(self, y):
        '''
        Returns the accuracy for the mini-batch
        '''
        return Tensor.mean(Tensor.eq(y , self.y_out))


# In[18]:

def size(data):
    '''
    Returns the size of the dataset 'data'
    '''
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    '''
    Drop out functionality using theano
    '''
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer * Tensor.cast(mask, theano.config.floatX)


import matplotlib.pyplot as plt

def plot_overlay(test_accuracy, validation_accuracy, x_label, x_units, y_min, y_max):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_units,
            [va for va in validation_accuracy], 'o-',
            color = '#2A6EA6', label="Best Accuracy on Validation Data")
    ax.plot(x_units,
            [ta for ta in test_accuracy], '^-',
            color='#FFA933', label="Best Accuracy on Test Data")
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_title('Best Accuracies on  Data')
    ax.set_ylim([y_min, y_max])
    plt.legend(loc='lower right')
    plt.show()

def plot_accuracy(accuracy, x_units, title, xlabel, ylabel, y_min, y_max):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([x for x in x_units],
            [a for a in accuracy], 'o-',
            color='#FFA933', label=ylabel)
    ax.set_ylim([y_min, y_max])
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.show()