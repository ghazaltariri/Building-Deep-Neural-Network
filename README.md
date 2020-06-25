# Building-Deep-Neural-Network
I will use two activation functions:

### Sigmoid
### ReLU

Building NN from scratch: 
These are the different methods for building NN:

1. initialize_parameters_deep(layer_dims) which returns parameters. We should make sure that dimensions match between each layes.
2. L_model_forward(X, parameters) which return AL, caches
3. compute_cost(AL, Y) which return cost
4. L_model_backward(AL, Y, caches) which return grads
5. update_parameters(parameters, grads, learning_rate) which return parameters
