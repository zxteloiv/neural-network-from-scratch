#!/usr/bin/env python2
# coding: utf-8

import random, math
from copy import copy
from naive_algebra import Vector, Matrix, dot_prod, mmul

class FeedForwardNetwork:
    """
    The basic feedforward network with basic back-propagation algorithm.
    """

    def __init__(self, layer_num, dim_list):
        """
        Constructor for network.
        Params:
        layer_num: the gross amount of layers, including input and output.
        dim_list: a list of the number of dimension for each layer.
        """
        if layer_num != len(dim_list):
            raise ValueError('All layer dimension should be provided')

        self.layer_num = layer_num

        self.dims = dim_list

        # if there are 3 layers, there will be only 2 matrix
        big_theta = [ Matrix.fromRandom(dim_list[layer_id], dim_list[layer_id - 1])
                for layer_id in xrange(layer_num - 1)]

        bias = [ Vector.fromRandom(dim_list[layer_id])
                for layer_id in xrange(layer_num - 1)]

    def inference(self, vector):
        vec = copy.copy(vector)
        for layer_id in xrange(self.layer_num - 1):
            theta = self.big_theta[layer_id]
            bias = self.bias[layer_id]

            vec = Vector.fromIterable(self.dims[layer_id],
                    (sigmoid(dot_prod(theta.row(neuron_id), vec) + bias[neuron_id])
                        for neuron_id in xrange(self.dims[layer_id + 1])))

        return vec

    def _forward(self, x, y):
        pass

    def _backward(self, x, y, activations):
        pass

    def train(self, sample_generator):
        for (x, y) in sample_generator():
            pass
        pass

def sigmoid(z):
    return 1.0 / (1 + math.exp(-z))

if __name__ == "__main__":
    print sigmoid(1)
    pass
