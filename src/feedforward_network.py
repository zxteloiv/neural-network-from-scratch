#!/usr/bin/env python2
# coding: utf-8

import random, math
from copy import copy
from naive_algebra import Vector, Matrix, dot_prod, mmul, vmul

class FeedForwardNetwork:
    """
    The basic feedforward network with basic back-propagation algorithm.
    """

    def __init__(self, dim_list, eta = 0.1):
        """
        Constructor for network.
        Params:
        dim_list: a list of the number of dimension for each layer.
        eta: learning rate for each gradient descent step
        """
        depth = len(dim_list)
        self.depth = depth
        self.dim_list = dim_list
        self.eta = eta

        # 1. Initiate each layer: output, partial_output and weight,
        #    although partial_output is useless for the input layer and weight is
        #    useless for the output layer.
        #
        # 2. Partial_weight is an internal variable and will not be stored in
        #    a layer.
        #
        self.layers = [ {'output':Vector.fromIterable(0 for i in dim_list[l]),
            'partial_output':Vector.fromIterable(0 for i in dim_list[l]),
            'weight':Matrix.fromRandom(dim_list[l], dim_list[l - 1] + 1)}
            for l in xrange(depth) ]

    def inference(self, vector):
        vec = copy.copy(vector)
        self._forward(self, vec)
        output = self.layers[self.depth - 1]['output']
        return Vector.fromIterable(1 if o > 0.5 else 0 for o in output.data)

    def _forward(self, x):
        # input layer
        self.layers[0]['output'].assign(x)

        # hidden layers and output layer
        for layer_id in xrange(1, self.depth):
            weight = self.layers[layer_id - 1]['weight']
            indata = self.layers[layer_id - 1]['output']
            self.layers[layer_id]['output'] = vsigmoid(vmul(weight, indata))

    def _backward(self, x, y):
        # output layer
        layer_id = self.depth - 1
        output_layer = self.layers[layer_id]
        output = output_layer['output']
        output_layer['partial_output'].assign(Vector.fromIterable(
            output[i] - y[i] for i in self.dim_list[layer_id]
            ))

        # hidden layer and input layer
        for layer_id in xrange(self.depth - 2, -1, -1):
            layer = self.layers[layer_id]
            weight = layer['weight']
            partial_output = layer['partial_output']
            last_output = self.layers[layer_id + 1]['output']
            last_partial = self.layers[layer_id + 1]['partial_output']

            """
            Partial output for every layer except the output one is:
            \frac {\partial E} {\partial O_k^{(l)}} =
                \sum_i (\frac {\partial E} {\partial O_i^{ (l+1) }}
                    * O_i^{ (l+1) } * (1 - O_i^{ (l+1) }) * w_{ik}^{ (l) } )
            
            But the partial output of the first layer is unnecessary,
            thus we don't compute it.
            """
            if layer_id > 0:
                layer['partial_output'].assign(Vector.fromIterable(
                    sum(last_partial[i] * last_output[i] * (1 - last_output[i])
                        * weight.item(i, k)
                        for i in self.dim_list[layer_id + 1])
                    for k in self.dim_list[layer_id] ))

            """
            Partial weight for every layer except the output one:
            \frac {\partial E} {\partial w_{ji}^{(l)}} = 
                \frac {\partial E} {\partial O_i^{(l + 1)}}
                    * O_j^{(l + 1)} * (1 - O_j^{(l+1)}) * O_i^{(l)}
            """
            partial_weight = Matrix.fromIterable(weight.row_num, weight.col_num, (
                    last_partial[row_id] 
                    * last_output[row_id] * (1 - last_output[row_id])
                    * output[col_id]
                    for row_id in self.dim_list[layer_id + 1]
                    for col_id in self.dim_list[layer_id]
                    ))

            weight -= self.eta * partial_weight

        pass

    def train(self, sample_generator):
        for (x, y) in sample_generator():
            pass
        pass

def sigmoid(z):
    return 1.0 / (1 + math.exp(-z))

def vsigmoid(v):
    return Vector.fromIterable(sigmoid(v[i]) for i in xrange(len(v)))

if __name__ == "__main__":
    print sigmoid(1)
    pass
