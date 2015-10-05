#!/usr/bin/env python2

#
# Using the same numeric example as in http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/.
# It's easy to check whether the code is correctly implemented.
#

import sys
import mnist_adapter
import feedforward_network
from itertools import izip

from naive_algebra import Vector, Matrix, dot_prod, mmul, vmul


import datetime

def puttime(msg):
    print datetime.datetime.now().strftime('%H:%M:%S ') + str(msg)

def main(mnist_path):
    puttime('start loading')

    network = feedforward_network.FeedForwardNetwork(
            dim_list = [2, 2, 2],
            eta = 0.5
            )

    network.layers[0]['weight'] = Matrix(2, 2, [.15, .20, .25, .30])
    network.layers[0]['bias'] = Vector.fromList([.35, .35])
    network.layers[1]['weight'] = Matrix(2, 2, [.40, .45, .50, .55])
    network.layers[1]['bias'] = Vector.fromList([.60, .60])

    x = Vector([.05, .10])
    y = Vector([.01, .99])

    def generator(x, y):
        yield (x, y)

    # start training
    puttime('start training')
    network.train(generator(x, y), puttime, limit = 1000)

if __name__ == '__main__':
    main('')

