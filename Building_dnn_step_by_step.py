import numpy as np
from dnn import sigmoid, sigmoid_backward, relu, relu_backward
np.random.seed(1)

def initialize_parameters_deep(layer_dims):
    """
    Arguments: layer_dims is a python array (list)
    Returns: parameters is a python dictionary
    """
    parameters = {}
    L = len(layer_dims) # number of layers in network
    
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['W'+str(l)].shape == (layer_dims[l],layer_dims[l-1]))
        assert(parameters['b'+str(l)].shape == (layer_dims[l],1))
        
    return (parameters)


def linear_forward(A, W, b):
    """
    Implement the linear part of the layer's forward propagation.
    Arguments: A is activations from previous layer
               W is weights matrix and
               b is bias vector
               
    Returns: Z is the input of activation function
             cache is a python dictionary containing "A", "W" and "b"(sorted for compuing the backward pass efficiently
    """
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A, W, b)

    return (Z, cache)


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the Linear->Activation layer
    Arguments: A_prev is activation from previous layer
               W is weights matrix
               b is bias vector
    Returns: A is the output of the activation function
             cache is a python dictionary containing "linear_cache" and "activation_cache" for computing the backward pass efficiency
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    
    return (A, cache)


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the 
    [Linear->Relu]*(L-1)->Linear->Sigmoid computation
    Arguments: X is a numpy array of shape
               parameters is the output of initialize_parameters_deep()
    Returns: AL is last post-activation value
             caches is list of caches containing every cache of linear_relu_forward() and the cache of linear_sigmoid_forward()
    """
    caches = []
    A = X
    L = len(parameters)//2 #number of layers in the neural network
    #Implement [Linear->Relu]*(L-1) and add cache to the caches list 
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters['W'+str(l)],
                                             parameters['b'+str(l)],
                                             activation='relu')
        caches.append(cache)
    #Implement Linear -> Sigmoid 
    AL, cache = linear_activation_forward(A,
                                          parameters['W'+str(l)],
                                          parameters['b'+str(l)],
                                          activation='sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    
    return(AL, caches)


def compute_cost(AL, Y):
    """
    Implement the cost function
    Arguments: AL is probability vector corresponding to label prediction
               Y is true label vector
               
    Returns: cost is cross-entropy cost
    """
    m = Y.shape[1]
    
    #Compute loss from AL and Y
    cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return (cost)


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation fo a single layer
    Arguments: dZ is gradient of the cost with respect to linear output
               cache is tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    Returns: dA_prev is gradient of the cost with respect to the activation of the previous layer
             dW is gradient of the cost with respect to W
             db is gradient of the cost with respect to b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, cache[0].T)/m
    db = np.squeeze(np.sum(dZ, axis=1, keepdims=True))/m
    dA_prev = np.dot(cache[1].T, dZ)
    
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(isinstance(db,float))
    
    return(dA_prev, dW, db)


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the Linear->Activation layer
    Arguments: dA is post-activation gradient for current layer
               cache is tuple of values we store for computing backward propagation efficiently
               activation is used in the layer as a text string:'sigmoid' or 'relu'
    Returns: dA_prev is gradient of the cost with respect to the activation
             dW is gradient of the cost with respect to W
             db is gradient of the cost with respect to b
    """
    linear_cache, activation_cache = cache
    
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return(dA_prev, dW, db)


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [Linear->Relu] * (L-1)->Linear->Sigmoid group
    Arguments: AL is probability vector
               Y is true label vector
               cache is list of caches
    Returns: grads is a dictionary with the gradients
             grads["dA"+str(l)], grads["dW"+str(l)],grads["db"+str(l)]
    """
    grads = {}
    L = len(caches) #the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    #Initializing the backpropagation
    dAL = -(np.divide(Y, AL)-np.divide(1-Y, 1-AL))
    
    current_cache = caches[-1]
    grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)] = linear_backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])
    
    for l in reversed(range(L-1)):
        
        current_cache = cache[l]
        dA_prev_temp, dW_temp, db_temp = linear_backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return(grads)


def update_parameters(parameters, grads, learning_rate):
    """
    Using gradient descent to update parameters
    Arguments: parameters is python dictionary containing the parameters
               grads is python dictionary containing gradients and the output of L_model_backward
    Returns: parameters that are updated
    """
    
    L = len(parameters)//2 #number of layers in the neural network
    
    for l in range(L):
        
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate* grads["dW" + str(l + 1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate* grads["db" + str(l + 1)]
        
    return(parameters)
        
        


        

