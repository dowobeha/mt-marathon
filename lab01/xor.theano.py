#!/usr/bin/env python3

import numpy
import theano
import theano.tensor as T
rng = numpy.random

X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
  [1, 1],
  [0, 1],
  [0, 0]
]

Y = [
  0, 1, 1, 0, 0, 1, 0
]

# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)

hidden = 2
w_1 = theano.shared(rng.randn(2,hidden), name="w_1")
w_2 = theano.shared(rng.randn(hidden), name="w_2")

# initialize the bias term
b_1 = theano.shared(rng.randn(hidden), name="b_1")
b_2 = theano.shared(0., name="b_2")

#print("Initial model:")
#print(w.get_value())
#print(b.get_value())

# Construct Theano expression graph
l_1 = 1 / (1 + T.exp(-T.dot(x, w_1) - b_1))
l_2 = 1 / (1 + T.exp(-T.dot(l_1, w_2) - b_2))   # Probability that target = 1
prediction = l_2 > 0.5                    # The prediction thresholded
xent = -y * T.log(l_2) - (1-y) * T.log(1-l_2) # Cross-entropy loss function
cost = xent.mean()                       # The cost to minimize


gw_1, gb_1, gw_2, gb_2 = T.grad(cost, [w_1, b_1, w_2, b_2])             
                                          
                                          # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

alpha = 0.1
                        
# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w_1, w_1 - alpha * gw_1), 
                   (b_1, b_1 - alpha * gb_1),
                   (w_2, w_2 - alpha * gw_2), 
                   (b_2, b_2 - alpha * gb_2)))

predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(10000):
    pred, err = train(X, Y)

#print("Final model:")
#print(w.get_value())
#print(b.get_value())
print("target values for X:")
print(Y)
print("prediction on X:")
print(predict(X))
