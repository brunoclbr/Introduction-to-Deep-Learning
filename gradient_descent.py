import numpy as np
import matplotlib.pyplot as plt
'''
This script minimizes the loss of a predictive linear funtction y_hat = w*x + b by means 
of gradient descent. We compute manually the gradient of the function with respect to the weights. 
'''

# initialize parameters
x = np.random.randn(10,1)
N = x.shape[0]
y = 2*x + np.random.rand()
w = 0.0
b = 0.0
alpha = 0.1 # learning rate

# create gradient descent function
def gradient_descent(x,y,w,b): 
  # L = (y-w*x - b)**2 is the loss function used for training, hence dldw and dldw
  dldw = 0.0
  dldb = 0.0
  
  for xi, yi in zip(x,y):
    dldw += -xi*2*(yi-w*xi - b)
    dldb += -2*(yi-w*xi - b)
  
  # I could also put this in the previous for loop increasing the computational demand and leaving out 1/N
  # this is how weight update is tipically done in major DL algorithms when data is divided into batches.
  # here the batch size equals the total size of the data
  w = w - alpha*(1/N)*dldw
  b = b - alpha*(1/N)*dldb
  return w, b

# apply algorithm
for epoch in range(20):
  ### FORWARD PASS ###
  y_hat = w*x + b
  mse = (1/N)*sum((y-y_hat)**2) # metric for accuracy evaluation, not for training
  ### BACKWARD PASS ###
  w, b = gradient_descent(x,y,w,b)

print(f'final loss {mse}, final parameters w = {w}, b = {b}')
plt.figure()
plt.plot(x,w*x + b,'r')
plt.plot(x,y,'b')
plt.show()