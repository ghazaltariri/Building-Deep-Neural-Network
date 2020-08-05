# Building-Deep-Neural-Network
I will use two activation functions:

### Sigmoid: ![imag] (https://latex.codecogs.com/svg.latex?%5Csigma%28Z%29%3D%5Csigma%28WA%2Bb%29%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%28WA%2Bb%29%7D%7D)
This function returns two items: the activation value "a" and a "cache" that contains "Z" 

### ReLU: The mathematical formula for ReLu is $A = RELU(Z) = max(0, Z)$

Building NN from scratch: 
These are the different methods for building NN:

1. initialize_parameters_deep(layer_dims) which returns parameters. We should make sure that dimensions match between each layes.
2. L_model_forward(X, parameters) which return AL, caches
3. compute_cost(AL, Y) which return cost
4. L_model_backward(AL, Y, caches) which return grads
5. update_parameters(parameters, grads, learning_rate) which return parameters

The model's structure is : [Linear-> Relu] X (L-1) -> Linear -> Sigmoid. I.e., it has L-1 layers using ReLU activation function followed by an output layer with a sigmoid activation function.

We can use random initialization for the weight metrics and zero initialization for biases.
After initialization, we will do the forward propagation module by implementing some basic functions:
-Linear
-Linear -> Activation where Activation will be either ReLU or Sigmoid
-[Linear-> Relu] X (L-1) -> Linear -> Sigmoid
