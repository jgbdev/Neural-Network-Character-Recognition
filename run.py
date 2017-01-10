#!/usr/bin/python
import numpy as np
import random as rand
import mninst_loader
import timeit
class Network(object):


    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y , x)
                        for x,y in zip (sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)

        return a

    def SGD(self, training_data, epochs , mini_batch_size, eta, test_data=None):
        """
        :param self:
        :param training_data: (X,Y) input.
        :param epochs:
        :param mini_batch_size: Subset size of training data
        :param eta: Number of iterations
        :param test_data:
        """

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
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):

        batch_size = len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]


        x_b = []
        y_b = []


        for x, y in mini_batch:
            x_b.append(x)
            y_b.append(y)

        delta_nabla_b, delta_nabla_w = self.backprop(x_b, y_b)

        for i in xrange(batch_size):
            for j in xrange(0,self.num_layers -1):
                nabla_b[j] += delta_nabla_b[j][i]
                nabla_w[j] += delta_nabla_w[j][i]

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        len_x = len(x)
        len_y = len(y)

        if(not (len_x == len_y)):
            print "Vectors x, y must be equal length"
            exit()


        nabla_b = [[np.zeros(b.shape) for _ in xrange(len_x)] for b in self.biases]
        nabla_w = [[np.zeros(w.shape) for _ in xrange(len_y)] for w in self.weights]

        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.einsum('ij,ajk->aik', w, activation) + [b for _ in xrange(len_x)]
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.einsum('aij,akj->aik', delta, activations[-2])

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.einsum('ji,ajk->aik', self.weights[-l + 1], delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.einsum('aij,akj->aik', delta, activations[-l - 1])
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


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def main():

    net = Network([784,15,10])
    training_data, validation_data, test_data = mninst_loader.load_data_wrapper()

    print "Starting"
    start_time = timeit.default_timer()
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    elapsed = timeit.default_timer() - start_time
    print "Elapsed time " + str(elapsed)



if __name__ == "__main__":
    main()
