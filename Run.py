import Network as n
import timeit
import mninst_loader as loader


def main():

    net = n.Network([784, 30, 10], cost=n.Network.CrossEntropy, debug=True)

    training_data, validation_data, test_data = loader.load_data_wrapper()

    print "Starting"
    start_time = timeit.default_timer()
    net.SGD(training_data, 30, 10, 0.1, decay=5.0, test_data=test_data, early_stop=10)
    elapsed = timeit.default_timer() - start_time
    print "Elapsed time " + str(elapsed)

if __name__ == "__main__":
    main()
