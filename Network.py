#!/usr/bin/python
from __future__ import division
import numpy as np
import random as rand
import mminst_loader
import math as m
import timeit


class Network(object):

    class QuadraticCost(object):

        @staticmethod
        def delta(z, a, y):
            return sigmoid_prime(z) * (a - y)

        @staticmethod
        def fn(a, y):
            return 0.5 * m.pow(y - a, 2)

    class CrossEntropy(object):

        @staticmethod
        def delta(z, a, y):
            return (a - y)

        @staticmethod
        def fn(a, y):
            return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    def __init__(self, sizes, cost=CrossEntropy, debug=False):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y , x)
                        for x,y in zip (sizes[:-1], sizes[1:])]
        self.cost = cost
        self.epoch_test_accuracy = []
        self.epoch_train_accuracy = []
        self.epoch_train_cost = []
        self.epoch_test_cost = []
        self.debug=debug
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a


    def weight_bias_initializer(self):


        return ""



    def SGD(self, training_data, epochs , mini_batch_size, eta, test_data=None):
        """
        :param self:
        :param training_data: (X,Y) input.
        :param epochs:
        :param mini_batch_size: Subset size of training data
        :param eta: Number of iterations
        :param test_data:
        """

        epoch_train_accuracy = []
        epoch_test_accuracy = []
        epoch_train_cost = []
        epoch_test_cost = []


        if test_data: n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):
            rand.shuffle(training_data)
            mini_batches = [
                training_data[k: k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches :
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                if self.debug:
                    print "Epoch {0}: {1} / {2}".format(
                        j, self.evaluate(test_data) , n_test)
            else:
                if self.debug:
                    print "Epoch {0} complete".format(j)

            epoch_train_accuracy.append(self.calc_accuracy(training_data, convert=True))
            epoch_test_accuracy.append(self.calc_accuracy(test_data))

            epoch_test_cost.append(self.calc_cost(test_data, convert=True))
            epoch_train_cost.append(self.calc_cost(training_data))

        self.epoch_train_accuracy = epoch_train_accuracy
        self.epoch_test_accuracy = epoch_test_accuracy
        self.epoch_train_cost = epoch_train_cost
        self.epoch_test_cost = epoch_test_cost

    def calc_accuracy(self, data, convert=False):

        if convert:
            test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                            for (x, y) in data]
        else:
            test_results = [(np.argmax(self.feedforward(x)), y)
                            for (x, y) in data]

        return sum(int(x == y) for (x, y) in test_results) / len(data)

    def calc_cost(self, data, convert=False):

        if convert:
            cost_total = [(self.cost).fn(self.feedforward(x), vectorized_result(y))
                          for (x,y) in data]
        else:
            cost_total = [(self.cost).fn(self.feedforward(x),y)
                            for (x , y) in data]

        return np.sum(cost_total)/len(data)



    def update_mini_batch(self, mini_batch, eta):


        x = np.column_stack((mini_batch[i][0] for i in xrange(len(mini_batch))))
        y = np.column_stack((mini_batch[i][1] for i in xrange(len(mini_batch))))

        nabla_b , nabla_w = self.backprop(x, y)

        for i in xrange(self.num_layers-1):
            nabla_b[i] = np.sum(nabla_b[i], axis=1).reshape(len(nabla_b[i]),1)

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + np.column_stack((b for _ in xrange(x.shape[1])))
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(z,activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e



def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def main():

    net = Network([784,100,10], cost=Network.CrossEntropy)

    training_data, validation_data, test_data = mminst_loader.load_data_wrapper()

    print "Starting"
    start_time = timeit.default_timer()
    net.SGD(training_data[1000], 30, 2, 0.5, test_data=test_data)
    elapsed = timeit.default_timer() - start_time
    print "Elapsed time " + str(elapsed)

if __name__ == "__main__":
    main()
