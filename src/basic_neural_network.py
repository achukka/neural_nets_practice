
# coding: utf-8

# In[33]:

import numpy as np
import random


# In[34]:

# Base Class of a Simple Neural Network
class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # np.radnom.rand is a gaussian with mean 0, variance 1
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x,y in zip(sizes[:-1], sizes[1:])]
        self.val=0
    # This is the feed forward operation of the net
    def feedforward(self, a):
        ''' Returns the output of the network. If 'a' is the input '''
        for b,w in zip(self.biases, self.weights):
            ''' a' = sigmoid(w.a + b) '''
            a = sigmoid(np.dot(w, a) + b)
        return a
    # Let us define the Stochastic Gradient Descent method.
    # The net uses SGD to learn the weights and biases
    def StochasticGradientDescent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        ''' 
            Trains the neural network using mini-batch stochastic gradient descent.
            The training data is list od tuples (x,y) representing the training input and desired outputs
            If test data is provided the network will be evaluated against after each epoch, 
            and partial progress is printed out. This is useful for tracking progress, 
            but things get slow substantially
        '''
        if test_data:
            n_test = len(test_data)
        n_train = len(training_data)
        for epoch in xrange(epochs):
            random.shuffle(training_data)
            ''' Get the list of examples to train '''
            mini_batches = [ training_data[it: it+mini_batch_size] 
                            for it in xrange(0, n_train, mini_batch_size)]
            ''' Train each mini batch '''
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} Complete".format(epoch)
                
    # We have to tell the network, how to update the weights
    def update_mini_batch(self, mini_batch, eta):
        ''' 
            The mini batch is a list of (x,y) tuples. eta is the learning rate
            For each tuple update the biases and weights using backpropagation
        '''
        
        # Initialize the partial derivates to zero
        Del_b = [np.zeros(b.shape) for b in self.biases]  # nala_b (partial derivates of b)
        Del_w = [np.zeros(w.shape) for w in self.weights] # nala_w (partial derivates of w)
        
        # For input pair calculate the derivates using back propagation
        for x, y in mini_batch:
            delta_Del_b, delta_Del_w = self.backprop(x, y)
            Del_b = [ nb + dnb for nb, dnb in zip(Del_b, delta_Del_b)]
            Del_w = [ nw + dnw for nw, dnw in zip(Del_w, delta_Del_w)]
        # Update final biases and weights
        self.weights = [w - (eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, Del_w) ]
        self.biases = [b - (eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, Del_b) ]
    
    # Back Propagation Algorithm
    # Takes as an input (x,y) tuple and outputs a tuple(nabla_b, nabla_w) of np arrays
    # nabla_b - the derivates of the cost function w.r.t the biases
    # nabla_w - the derivates of the cost function w.r.t the weights
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feedforward
        activation = x    # Activation of the input layer
        activations = [x] # Store the list of activations layer by layer
        latents = []      # store the list of latent variables z
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            latents.append(z)
            activation = sigmoid(z)        
            activations.append(activation)
        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(latents[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Now back propagate the error through all the layers
        # Note that for the last layer, l = self.num_layers - 1, 
        # so we start from self.num_layers - 2 till layer 1 (i.e the second layer)
        for l in xrange(2, self.num_layers):
            z = latents[-l]
            ap = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * ap
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    # Define cost function for the neural network
    def cost(self, output_activations, y):
        return 0.5* np.linalg.norm(output_activations-y)**2
    
    # Define the cost derivate for the output unit
    # Returns a vector of partial derivates (\partial C_x/ \partial a) 
    # for ouptut activation units
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
    # Evaluate the test data. The network output is assumed to be 
    # the index of neuron that has the highest activation
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)    
# Let us define the activation function - sigmoid are also known as logistic units
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))