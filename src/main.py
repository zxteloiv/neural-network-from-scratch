#!/usr/bin/env python2

import sys
import mnist_adapter
import feedforward_network
from itertools import izip

import datetime

def puttime(msg):
    print datetime.datetime.now().strftime('%H:%M:%S ') + str(msg)

def main(mnist_path):
    puttime('start loading')

    loader = mnist_adapter.MNIST(mnist_path)
    training_data = loader.load_training()
    testing_data = loader.load_testing()

    input_size = len(training_data[0][0])

    network = feedforward_network.FeedForwardNetwork(
            dim_list = [input_size, 300, 10],
            eta = 0.1
            )

    # start training
    puttime('start training')
    for i in xrange(1):
        generator = feedforward_network.sample_wrapper(training_data)
        network.train(generator, puttime, limit = 1000)

    # start testing
    puttime('start testing')
    generator = feedforward_network.sample_wrapper(testing_data)
    succ, total = 0, 0
    for (x, y) in generator:
        out = network.inference(x)
        print "final", out, "label", y
        succ += (1 if out == y else 0)
        total += 1
        if total >= 50: break

    puttime('end testing')
    print "Prec = %.4f" % (succ * 1.0 / total)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print "Usage: %s mnist_path" % sys.argv[0]

