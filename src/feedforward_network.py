#!/usr/bin/env python2
# coding: utf-8

import random, math
from copy import copy
from naive_algebra import Vector, Matrix, dot_prod, mmul, vmul
from itertools import izip

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
        #    although partial_output is useless for the input layer, similarly
        #    weight and bias are useless for the output layer.
        #
        # 2. Partial_weight is an internal variable and will not be stored in
        #    a layer.
        #
        self.layers = [ {'output':Vector.fromIterable(0 for i in xrange(dim_list[l])),
            'partial_output':Vector.fromIterable(0 for i in xrange(dim_list[l])),
            'weight':Matrix.fromRandom(dim_list[l + 1], dim_list[l]),
            'bias':Vector.fromRandom(dim_list[l + 1])}
            for l in xrange(depth - 1) ]
        
        # output layer
        self.layers.append({'output':Vector.fromList([0] * dim_list[depth - 1]),
            'partial_output':Vector.fromList([0] * dim_list[depth - 1]),
            'weight': None, 'bias': None})

    def inference(self, vector):
        self._forward(vector)
        output = self.layers[self.depth - 1]['output']
        maxpos, maxval = 0, 0
        for i in xrange(len(output)):
            if maxval < output[i]:
                maxpos, maxval = i, output[i]
        rtn = Vector.fromList([0] * len(output))
        rtn[maxpos] = 1
        return rtn

    def _forward(self, x):
        # input layer
        self.layers[0]['output'].assign(x)

        # hidden layers and output layer
        for layer_id in xrange(1, self.depth):
            weight = self.layers[layer_id - 1]['weight']
            indata = self.layers[layer_id - 1]['output']
            bias = self.layers[layer_id - 1]['bias']
            output = (vmul(weight, indata) + bias)
            # normalize with the amount of neurons in last layer
            output = vsigmoid(output / (self.dim_list[layer_id - 1] + 1.0))
            self.layers[layer_id]['output'] = output

    def _backward(self, x, y):
        # output layer
        layer_id = self.depth - 1
        output = self.layers[layer_id]['output']
        self.layers[layer_id]['partial_output'].assign(Vector.fromIterable(
            output[i] - y[i] for i in xrange(self.dim_list[layer_id])
            ))

        loss = sum((output[i] - y[i]) ** 2 for i in xrange(self.dim_list[layer_id]))

        # hidden layer and input layer
        for layer_id in xrange(self.depth - 2, -1, -1):
            weight = self.layers[layer_id]['weight']
            bias = self.layers[layer_id]['bias']
            partial_output = self.layers[layer_id]['partial_output']
            output = self.layers[layer_id]['output']
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
                self.layers[layer_id]['partial_output'].assign(Vector.fromIterable(
                    sum(last_partial[i] * last_output[i] * (1 - last_output[i])
                        * weight.item(i, k)
                        for i in xrange(self.dim_list[layer_id + 1]))
                    / (self.dim_list[layer_id] + 1.0)
                    for k in xrange(self.dim_list[layer_id] )))

            """
            Partial weight for every layer except the output one:
            \frac {\partial E} {\partial w_{ji}^{(l)}} = 
                \frac {\partial E} {\partial O_j^{(l + 1)}}
                    * O_j^{(l + 1)} * (1 - O_j^{(l+1)}) * O_i^{(l)}
            """
            partial_weight = Matrix.fromIterable(weight.row_num, weight.col_num, (
                    last_partial[row_id] 
                    * last_output[row_id] * (1 - last_output[row_id])
                    * output[col_id]
                    / (self.dim_list[layer_id] + 1.0)
                    for row_id in xrange(self.dim_list[layer_id + 1])
                    for col_id in xrange(self.dim_list[layer_id])
                    ))

            self.layers[layer_id]['weight'] -= self.eta * partial_weight

            """
            Partial bias is almost exact as the partial weight,
            but for every item in the bias vector the last item is 1
            \frac {\partial E}{\partial b_j^{(l)}} =
                \frac {\partial E}{\partial O_j^{(l + 1)}}
                    O_j^{(l + 1)} (1 - O_j^{(l+1)}) * 1
            """
            partial_bias = Vector.fromIterable(
                    last_partial[row_id]
                    * last_output[row_id] * (1 - last_output[row_id]) * 1
                    / (self.dim_list[layer_id] + 1.0)
                    for row_id in xrange(self.dim_list[layer_id + 1])
                    )
            self.layers[layer_id]['bias'] -= self.eta * partial_bias

        return loss

    def train(self, generator, logger = None, limit = 100):
        counter = 0
        for (x, y) in generator:
            self._forward(x)
            loss = self._backward(x, y)

            if logger != None and counter % 10 == 0:
                logger(str(counter) + "," + str(loss))

            counter += 1
            if counter >= limit: break

def sigmoid(z):
    return 1.0 / (1 + math.exp(-z))

def vsigmoid(v):
    return Vector.fromIterable(sigmoid(v[i]) for i in xrange(len(v)))

def sample_wrapper(data):
    for (img, label) in izip(data[0], data[1]):
        x = Vector(img)
        y = Vector.fromIterable(1 if pos == label else 0 for pos in xrange(10))
        yield (x, y)

if __name__ == "__main__":
    print sigmoid(1)
    pass
