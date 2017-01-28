import Network as n1
import Network2 as n2
import timeit




def main():



    training_data, validation_data, test_data = n2.load_data_shared()
    mini_batch_size = 20
    # net = n1.Network([784, 30, 10], cost=n1.Network.CrossEntropy, debug=True)
    net = n2.Network([n2.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
              n2.ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
              n2.FullyConnectedLayer(n_in=40*4*4, n_out=100),
              n2.FullyConnectedLayer(n_in=100, n_out=100),
              n2.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    '''
    net = n2.Network([
        n2.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
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