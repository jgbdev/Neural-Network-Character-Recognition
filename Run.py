import Network as n1
import Network2 as n2
import mminst_loader
import timeit




def main():

    net2 = n2.FullyConnectedLayer(200,100)

    net = n1.Network([784, 30, 10], cost=n1.Network.CrossEntropy, debug=True)

    training_data, validation_data, test_data = mminst_loader.load_data_wrapper()

    print "Starting"
    start_time = timeit.default_timer()
    net.SGD(training_data, 30, 10, 0.1, decay=5.0, test_data=test_data, early_stop=10)
    elapsed = timeit.default_timer() - start_time
    print "Elapsed time " + str(elapsed)

if __name__ == "__main__":
    main()