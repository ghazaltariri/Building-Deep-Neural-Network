from Building_dnn_step_by_step import *
import numpy as np


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments: X is the data, numpy array of shape (number of examples, num_px * num_px * 3)
               Y is true "label" vector of shape (1, number of examples)
               layers_dims is list containing the input size and each layer size, of length (number of layers + 1).
               learning_rate is learning rate of the gradient descent update rule
               num_iterations is number of iterations of the optimization loop
       
    Returns:   parameters is parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
            

    return parameters
