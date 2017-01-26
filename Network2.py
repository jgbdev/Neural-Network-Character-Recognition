import theano
import numpy as np


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=2, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

        self.w = theano.shared(
            np.asarray(np.random.normal(
                loc=0.0,scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)

        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]



    def set_inpt(self, inpt, inpt_dropout, minibatch_size):
        self.inpt = inpt.reshape((minibatch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        )
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout =


