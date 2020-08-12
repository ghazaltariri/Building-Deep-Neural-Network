import numpy as np

def sigmoid(Z):
    
    """
    Sigmoid Activation in numpy
    input:
    Z: numpy array of any shape
    output:
    A: sigmoid(Z), same shape as Z
    cache: returns Z as well, usefull for backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A,cache


def relu(Z):
    """
    RELU Activation function
    input:
    Z: numpy array of any shape
    output:
    A: Post-activation parameter, of the same shape as Z
    cache: a python dictionary containing "A"
    stored for computing the backward pass efficiently
    """
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    
    return A,cache


def relu_backward(dA,cache):
    """
    The backward propagation for a single RELU unit
    input:
    dA: Post-activation gradient, of any shape
    cache: 'Z' where we store for computing backward propagation efficiently
    output:
    dZ: gradient of the cost with respect to Z 
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    
    return dZ


def sigmoid_backward(dA, cache):
    """
    The backward propagation for a single SIGMOID unit
    input:
    dA: post Activation gradient, of any shape
    cache: Z where we store for computing backward propagation efficiently
    output:
    dZ: Gradient of the cost with respect to Z
    """
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA*s*(1-s)
    
    assert(dZ.shape == Z.shape)
    
    return dZ
    