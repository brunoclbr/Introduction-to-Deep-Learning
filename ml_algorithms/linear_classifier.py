# Generate synthetic data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create 2D data with same covariance but different mean so that they are easily separable

num_samples = 1000

# inputs as 2D data!! mean has a value in x and y direction
first_class = np.random.multivariate_normal(mean=[3, 0],
                                            cov=[[1, 0.5], [0.5, 1]], size=num_samples) #1000,2
second_class = np.random.multivariate_normal(mean=[0, 3],
                                            cov=[[1, 0.5], [0.5, 1]], size=num_samples) #1000,2

stacked_inputs = np.vstack(((first_class,second_class))).astype(np.float32)#, 2000,2
#print(stacked_inputs)
# labels
targets = np.vstack(
    (np.zeros((num_samples,1),dtype="float32"),
     np.ones((num_samples,1),dtype="float32")) )
# visualize data
plt.figure()
plt.scatter(stacked_inputs[:,0], stacked_inputs[:,1], c=targets)

input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))
alpha = 0.1 #learning rate

#forward pass
def model(inputs):
  return tf.matmul(inputs, W) + b

#backward pass
def my_loss(x,y):
  square_loss = tf.square(x-y)
  return tf.reduce_mean(square_loss)

def backward_pass(training_data):
   # if I dont take the mean, output is shape(2000,1)
  with tf.GradientTape() as tape:
    prediction = model(stacked_inputs)
    loss = my_loss(prediction, targets)
    ''' The prediction = model(stacked_inputs) line needs to be within the tf.GradientTape() block because the
    prediction tensor is used in the computation of the loss. TensorFlow needs to record the operations involving
    prediction to calculate the gradient of the loss with respect to the trainable variables (W and b). If this
    line is outside the tf.GradientTape() block, TensorFlow won't track the operations, and when you try to
    compute the gradient later, it might not have the necessary information to do so.
    '''
  gradW, gradb = tape.gradient(loss, [W,b])
  W.assign_sub(gradW*alpha)
  b.assign_sub(gradb*alpha)
  return loss

# epochs iteration
for epoch in range(30):
  loss = backward_pass(stacked_inputs)
  print(f'Loss at epoch {epoch}: {loss:.4f}')

  predictions = model(stacked_inputs)
#plt.scatter(stacked_inputs[:, 0], stacked_inputs[:, 1], c=predictions[:, 0] > 0.5)
x = np.linspace(-3, 6, 100)
y = - W[0] /  W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(stacked_inputs[:, 0], stacked_inputs[:, 1], c=predictions[:, 0] > 0.5)