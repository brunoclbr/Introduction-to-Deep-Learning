import numpy as np
'''
This code goes through the actual back-propagation algorithm and the application of the chain-rule
with the input tensors/matrixes. Within tensorflow this step is automatically done with the gradient.tape()
method. 
'''

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, size=(100, 1))

# Initialize weights and biases
input_size = X.shape[1]
hidden_size = 3
output_size = 1

weights_input_hidden = np.random.rand(input_size, hidden_size)#(2,3), 6 weights
weights_hidden_output = np.random.rand(hidden_size, output_size) #(3,1) 3 weights

bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))

# Set hyperparameters
learning_rate = 0.01
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden #Hin = X*Wi + B1 (100,3)
    if epoch == 0:
      print(f' dot product X*weights_i_hidden{np.dot(X, weights_input_hidden)}')
    hidden_layer_output = sigmoid(hidden_layer_input) # Hout = s(Hin) (100,3)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output #Yin = Y*Wo + B2 (100,1)
    predicted_output = sigmoid(output_layer_input) #Yout = s(Yin)

    # Compute mean squared error
    loss = np.mean((y - predicted_output) ** 2)

    # Backward pass (Gradient Descent)
    output_error = y - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output) #

    hidden_layer_error = output_delta.dot(weights_hidden_output.T)# (Yout-y)*s'(Yout)*(Whidden)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += learning_rate * hidden_layer_output.T.dot(output_delta)
    bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

    weights_input_hidden += learning_rate * X.T.dot(hidden_layer_delta)
    bias_hidden += learning_rate * np.sum(hidden_layer_delta, axis=0, keepdims=True)

    # Print the loss at certain intervals
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Print the final weights and biases
print("Final Weights (Input to Hidden):")
print(weights_input_hidden)
print("Final Biases (Hidden):")
print(bias_hidden)
print("Final Weights (Hidden to Output):")
print(weights_hidden_output)
print("Final Biases (Output):")
print(bias_output)
