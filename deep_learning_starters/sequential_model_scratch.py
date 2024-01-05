import tensorflow as tf
import numpy as np
import math

'''
In order to understand how neural networks work on the inside, this script shows how to create, chain and
implement dense layers into a trainable model from scratch.  
'''

class NaiveDenseLayer():
  ''' Implements a dense layer step by step using the following 
  input transformation: output = activation_func(dot(W, input) + b) 
  '''
  def __init__(self, input_size, output_size, activation):
    self.activation = activation

    w_shape = (input_size, output_size)
    w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
    # create a variable object W representing weight matrix with randmonly initialized values
    self.W = tf.Variable(w_initial_value)
    # TensorFlow functions or operations might expect shapes to be represented as tuples, even if they have only one dimension
    b_shape = (output_size,) 
    b_initial_value = tf.zeros(b_shape)
    self.b = tf.Variable(b_initial_value)

  def __call__(self, inputs):
    return self.activation(tf.matmul(inputs,self.W) + self.b) # forward pass when calling object

  #the @property decorator is used to define a method that can be accessed like an attribute, 
  #providing a way to encapsulate the implementation details of a property while exposing a clean interface to the outside
  @property
  def weights(self):
    return [self.W, self.b]

class NaiveSequential():
  ''' class for chaining layers '''

  def __init__(self, layers):
    #The layers will be defined as a list of layers that will be passed as
    #model = NaiveSequential(layers=[layer1, layer2, layer3])
    self.layers = layers

  def __call__(self, inputs):
    # When calling the object a batch of inputs will be passed through the layers output_tensor = model(inputs)
    x = inputs
    for layer in self.layers:
      x = layer(x) # one transformation will be applied and then stored in x, then go to the next layer
    return tf.Variable(x)

  @property
  def weights(self):
    flat_weights = []
    for layer in self.layers:
        for w in layer.weights:
            flat_weights.append(tf.reshape(w, shape=(-1,)))
    return tf.Variable(tf.concat(flat_weights, axis=0))

class BatchGenerator():
  ''' create class for separating data into batches '''
  def __init__(self, images, labels, batch_size=128):
    assert len(images) == len(labels)
    self.index = 0
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    self.num_batches = math.ceil(len(images) / batch_size)

  def next(self):
    images = self.images[self.index : self.index + self.batch_size]
    labels = self.labels[self.index : self.index + self.batch_size]
    self.index += self.batch_size
    return images, labels

def update_weights(gradients, weights, lr=1e-3):
  for g,w in zip(gradients, weights):
    w.assign_sub(g*lr) # w = w -alpha*nabla(L) w.r.t w

# training step
def one_training_step(model, images_batch, labels_batch):
  ''' run the forward pass '''
  with tf.GradientTape() as tape:
    #tape.watch(model.weights)
    predictions = model(images_batch)
    per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
        labels_batch, predictions)
    print("Per sample losses:", per_sample_losses.shape)
    average_loss = tf.reduce_mean(per_sample_losses)
  print(model.weights)
  gradients = tape.gradient(average_loss, model.weights) # del(L)/del(w)
  print(f'Gradients are {gradients}')
  # apply back propagation
  update_weights(gradients, model.weights)

# full training loop
def fit(model, images, labels, epochs, batch_size=128):
  for epoch_counter in range(epochs):
    print(f'Epoch: {epoch_counter}')
    batch_generator = BatchGenerator(images, labels)
    for batch_counter in range(batch_generator.num_batches):
      images_batch, labels_batch = batch_generator.next() # will update the indexing of the batches automatically
      loss = one_training_step(model, images_batch, labels_batch) #per batch ONE weight update in order to save memory
      # that's why average loss
      if batch_counter % 100 == 0:
        print(f'loss at batch {batch_counter}: {loss:.2f}') # .2f: This is a formatting specification. :: Separates the variable/expression from the formatting instructions..2:
        # Specifies the number of digits to the right of the decimal point. f: Indicates that the variable/expression should be formatted as a floating-point number.

# Mock Keras Model. Makes sure that size of output hidden layer = size input output layer
model = NaiveSequential([
    NaiveDenseLayer(input_size=28*28, output_size=512, activation=tf.nn.relu),
    NaiveDenseLayer(input_size=512, output_size=10, activation=tf.nn.softmax) 
])

# import MNIST data from TF 
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
#print((train_images!=0).any())
fit(model, train_images, train_labels, epochs=5, batch_size=128)

predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")