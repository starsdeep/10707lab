from builtins import range
from builtins import object
import numpy as np
import sys
from cs231n.layers import *
from cs231n.layer_utils import *
import math


class Autoencoder(object):
    def __init__(self, input_dim=28*28, hidden_dim=100,
                 weight_scale=1e-1, reg=0.0, dropout_level=None):


        self.params = {}
        self.reg = reg
        self.dropout_level = dropout_level

        W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)
        W2 = W1.T # W2 is a reference to W.T
        b2 = np.zeros(input_dim)

        self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


    def loss(self, X, y=None):
        loss, grads = 0, {}
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], \
                self.params['W2'], self.params['b2']

        X_in = X if self.dropout_level==None else self.dropout(X, self.dropout_level)
        h1, cache_h1 = affine_sigmoid_forward(X_in, W1, b1)
        h2, cache_h2 = affine_sigmoid_forward(h1, W2, b2)
        data_loss, dh2 = l2_loss(h2, X)
        #add l2 regularization
        reg_loss = 0.5 * self.reg * np.sum(W1**2)
        loss = data_loss + reg_loss

        dh1, dW2, grads['b2'] = affine_sigmoid_backward(dh2, cache_h2)
        dx, dW1, grads['b1'] = affine_sigmoid_backward(dh1, cache_h1)
        
        grads['W2'] = (dW2 + dW1.T) / 2 
        grads['W1'] = (dW1 + dW2.T) / 2 

        #add L2 regularization gradient
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2'] 

        return loss, grads


    def reconstruct_error(self, X):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], \
                self.params['W2'], self.params['b2']

        X_in = X if self.dropout_level==None else self.dropout(X, self.dropout_level)        
        h1, cache_h1 = affine_sigmoid_forward(X_in, W1, b1)
        h2, cache_h2 = affine_sigmoid_forward(h1, W2, b2)
        data_loss, dh2 = l2_loss(h2, X)
        return data_loss

    def dropout(self, x, dropout_level):
        mask = np.random.binomial(size=x.shape, n=1, p = 1 - dropout_level)
        return mask * x
