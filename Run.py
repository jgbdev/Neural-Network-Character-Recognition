import Network as n1
import Network2 as n2
import timeit
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time



def checkGPU():

    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')


def main():

    checkGPU()

    training_data, validation_data, test_data = n2.load_data_shared()
    mini_batch_size = 20
    # net = n1.Network([784, 30, 10], cost=n1.Network.CrossEntropy, debug=True)
    net = n2.Network([n2.ConvPoolLayer(input_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
              n2.ConvPoolLayer(input_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
              n2.FullyConnectedLayer(n_in=40*4*4, n_out=100),
              n2.FullyConnectedLayer(n_in=100, n_out=100),
              n2.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    '''
    net = n2.Network([
        n2.ConvPoolLayer(input_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        n2.FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
        n2.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    '''
    print "Starting"
    start_time = timeit.default_timer()
    net.SGD(training_data,60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
    #net.SGD(training_data, 30, 10, 0.1, decay=5.0, test_data=test_data, early_stop=10)
    elapsed = timeit.default_timer() - start_time
    print "Elapsed time " + str(elapsed)

if __name__ == "__main__":
    main()