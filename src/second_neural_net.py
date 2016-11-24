
# coding: utf-8

# In[18]:

# Numpy - A Linear Algebra library in python
import numpy as np
# Matplotlib - A Library for plots
import matplotlib.pyplot as plt


# In[19]:

# Import system libraries
import json
import sys
import random


# In[20]:

# Qudratic Cost function
class QuadtricCost(object):
    
    @staticmethod
    def fn(a, y):
        ''' Returns the cost associated with predicted output 'a' 
            and desired output 'y'
        '''
        return 0.5*np.linalg.norm(a-y)**2
    
    @staticmethod
    def delta(z, a, y):
        ''' Returns the error delta from the output layer
        '''
        return (a-y)*sigmoid_prime(z)    


# In[21]:

# Cross Entropy Cost function
class CrossEntropyCost(object):
    
    @staticmethod
    def fn(a, y):
        ''' Returns the cost associated with predicted output 'a' 
            and desired output 'y'
            If a=1.0 and y=1.0 then (1-y)np.log(1-a) returns nan
            np.nan_to_num ensures that the output is 0.0
        '''
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z, a, y):
        ''' Returns the error delta from the output layer
            Here we do not use value z. 
            Rather it is kept for the consistency of the interface
        '''
        return (a-y)


# In[46]:

# Neural Network
class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        ''' List sizes contains the number of neurons in their respective layers
             The biases and weights are initialzed using default weight intializer
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_intializer()
        self.cost = cost
        
    def default_weight_intializer(self):
        ''' Standard Initilzation for weights.
             Weights are intialized using a Gaussian Distribution of mean 0 and variance 1
             Biases are intialized using a Gaussian Distribution of mean 0 and variance 1
        '''
        # np.radnom.rand is a gaussian with mean 0, variance 1
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/ np.sqrt(x)
                        for x,y in zip(self.sizes[:-1], self.sizes[1:])]
        
    def large_weigts_initializer(self):
        ''' Standard Initilzation for weights.
             Weights are intialized using a Gaussian Distribution of mean 0 and variance 1
             over the square root of weights connecting to the same neuron
             Biases are intialized using a Gaussian Distribution of mean 0 and variance 1
        '''
        # np.radnom.rand is a gaussian with mean 0, variance 1
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x,y in zip(self.sizes[:-1], self.sizes[1:])]

    def my_weights_intializer(self):
        ''' My Initilzation for weights.
             Weights are intialized using a Gaussian Distribution of mean 0 and variance 1
             over the log of weights connecting to the same neuron
             Biases are intialized using a Gaussian Distribution of mean 0 and variance 1
        '''
        # np.radnom.rand is a gaussian with mean 0, variance 1
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/ np.log(x)
                        for x,y in zip(self.sizes[:-1], self.sizes[1:])]
        
    # This is the feed forward operation of the net
    def feedforward(self, a):
        ''' Returns the output of the network. If 'a' is the input '''
        for b,w in zip(self.biases, self.weights):
            ''' a' = sigmoid(w.a + b) '''
            a = sigmoid(np.dot(w, a) + b)
        return a
    # Let us define the Stochastic Gradient Descent method.
    # The net uses SGD to learn the weights and biases
    def StochasticGradientDescent(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0, 
                                  evaluation_data=None, monitor_evaluation_cost=False,
                                  monitor_evaluation_accuracy=False, monitor_training_cost=False,
                                  monitor_training_accuracy=False):
        ''' 
            Trains the neural network using mini-batch stochastic gradient descent.
            The training data is list a tuples (x,y) representing the training input and desired outputs
            If evaluation data is provided the network will be evaluated against after each epoch,
            It also monitors the cost and accuracy for training and evaluation using appropriate flags,
            evaluated/calculated per epoch. The method would finally returns these four lists.
            Remember the lists would be empty if the flags are not set.
        '''
        if evaluation_data:
            n_eval = len(evaluation_data)
        n_train = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for epoch in xrange(epochs):
            random.shuffle(training_data)
            ''' Get the list of examples to train '''
            mini_batches = [ training_data[it: it+mini_batch_size] 
                            for it in xrange(0, n_train, mini_batch_size)]
            ''' Train each mini batch '''
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n_train)
            if epoch %10==0:
                print 'Epoch {0}: training complete'.format(epoch)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                if epoch%10==0:
                    print 'Training cost is:',cost
            if monitor_training_accuracy:
                accuracy = round(( 1.0 * self.accuracy(training_data, convert=True)/n_train), 3)
                training_accuracy.append(accuracy)
                if epoch%10==0:
                    print 'Training Accuracy is:',accuracy
            if evaluation_data and monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                if epoch %10==0:
                    print 'Evaluation cost is:',cost
            if evaluation_data and monitor_evaluation_accuracy:
                accuracy = round((1.0 * self.accuracy(evaluation_data)/n_eval), 3)
                evaluation_accuracy.append(accuracy)
                if epoch %10==0:
                    print 'Evaluation Accuracy is:',accuracy
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
                
    # We have to tell the network, how to update the weights
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
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
        # w_new = ( 1 - eta*lambda/n)*w - eta/len(mini_batch)*nw
        self.weights = [(1 - eta*(lmbda/n))*w - (eta/len(mini_batch))*nw 
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
        delta = (self.cost).delta(latents[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Now back propagate the error through all the layers
        # Note that for the last layer, l = self.num_layers - 1, 
        # so we start from self.num_layers - 2 till layer 1 (i.e the second layer)
        for l in xrange(2, self.num_layers):
            z = latents[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    
    def total_cost(self, data, lmbda, convert=False):
        '''
        Returns the total cost with Cross Entropy Cost and L2 Regularization.
        Covert should be set to 'False' for training data and 'True' for evaluation data
        Since we have not converted data to vectors in MNIST Data
        '''
        cost = 0.0
        for x,y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a,y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    def total_cost_l1_regularization(self, data, lmbda, convert=False):
        '''
        Returns the total cost with Cross Entropy Cost and L1 Regularization.
        Covert should be set to 'False' for training data and 'True' for evaluation data
        Since we have not converted data to vectors in MNIST Data
        '''
        cost = 0.0
        for x,y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a,y)/len(data)
        cost += 0.5*lamda/len(data) *sum(np.abs(w) for w in self.weights)
        return cost
    
    def accuracy(self, data, convert=False):
        '''
        Returns the accuracy of the model. ie the number of inputs for which the 
        expected output is same as the actual output.
        'Convert' should be set to 'False' for evaluation data and 
        'True' for training data.
        '''
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x,y in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for x,y in data]
        return sum(int(x==y) for x,y in results)
    
    # Saving a network to file
    def save(self, filename):
        '''
        Saves an instance of network into the file using the filename
        '''
        data = {"sizes": self.sizes,  "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases], "cost":str(self.cost.__name__)}
        fp = open(filename, "w")
        json.dump(data, fp)
        fp.close()


# In[47]:

# Miscellaneous Functions #
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# Derivative of the logistic function
def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))

def vectorized_result(d):
    ''' Returns a 10 dimensional vector with dth dimnesion set to 1.0 and rest are zero
    '''
    vec = np.zeros((10, 1))
    vec[d] = 1.0
    return vec


# In[48]:

# Loading a Network
def load(filename):
    '''
    Load a neural network from the file using the filename.
    Returns an instance of the network
    '''
    fp = open(filename, 'r')
    data = json.load(fp)
    fp.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs),
            training_cost[training_cost_xmin:num_epochs],
            color = '#2A6EA6')
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on Training Data')
    plt.show()


# In[51]:

def plot_test_cost(test_cost, num_epochs, test_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_cost_xmin, num_epochs),
            test_cost[test_cost_xmin:num_epochs],
            color = '#2A6EA6')
    ax.set_xlim([test_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on Test Data')
    plt.show()


# In[70]:

def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs),
            [accuracy for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
            color = '#2A6EA6')
    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy(%) on Test Data')
    plt.show()


# In[66]:

def plot_train_accuracy(train_accuracy, num_epochs, train_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(train_accuracy_xmin, num_epochs),
            [accuracy for accuracy in train_accuracy[train_accuracy_xmin:num_epochs]],
            color = '#2A6EA6')
    ax.set_xlim([train_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy(%) on Training Data')
    plt.show()


def plot_overlay(test_accuracy, train_accuracy, num_epochs, xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs),
            [accuracy for accuracy in train_accuracy[xmin:num_epochs]],
            color = '#2A6EA6', label="Accuracy on Training Data")
    ax.plot(np.arange(xmin, num_epochs),
            [accuracy for accuracy in test_accuracy[xmin:num_epochs]],
            color='#FFA933', label="Accuracy on Test Data")
    ax.set_xlim([xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy(%) on Training Data')
    ax.set_ylim([0.90, 1.00])
    plt.legend(loc='lower right')
    plt.show()

# Plotting overlay accuracy values with varying factors
def plot_overlay_accuracy_values(accuracy_values, varying_factors, varying_factor_name, 
                                 title, num_epochs, xmin, ymin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for index in xrange(len(accuracy_values)):
        ax.plot(np.arange(xmin, num_epochs),
                [accuracy_value for accuracy_value in accuracy_values[index][xmin:num_epochs]],
                label=varying_factor_name+":"+str(varying_factors[index]))
    ax.set_xlim([xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title(title)
    ax.set_ylim([ymin, 1.00])
    plt.legend(loc='lower right')
    plt.show()
    
# Plotting overlay accuracy values with varying factors
def plot_overlay_cost_values(cost_values, varying_factors, varying_factor_name, 
                                 title, num_epochs, xmin, ymin, ymax):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for index in xrange(len(cost_values)):
        ax.plot(np.arange(xmin, num_epochs),
                [cost_value for cost_value in cost_values[index][xmin:num_epochs]],
                label=varying_factor_name+":"+str(varying_factors[index]))
    ax.set_xlim([xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title(title)
    ax.set_ylim([ymin, ymax])
    plt.legend(loc='best')
    plt.show()