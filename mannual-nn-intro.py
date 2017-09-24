"""
  Inputs	   |  Output
0	0	1  |   0
0	1	1  |   0
1	0	1  |   1
1	1	1  |   1

use 2 layer l0 (input), l1 (hidden layer) with bridge of syn0
"""



import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])   #(4x3)

# output dataset            
y = np.array([[0,0,1,1]]).T   #(1x4)

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
# random 3x1 array of random numbers from [-1,1)
syn0 = 2*np.random.random((3,1)) - 1    #(3x1)

for iter in range(10000):

    # forward propagation
    l0 = X # train based on same sample input each time
    l1 = nonlin(np.dot(l0,syn0)) # (4x3) dot (3x1) = (4x1)
    # how much did we miss?
    l1_error = y - l1   #labeled - predicted: (4x1)-(4x1)

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True) # error * predicted's slope, if predict value is confident(-1,1 from sigmoid function), slope will close to 0, error then will be small, otherwise error * slope close to 1

    # update weights
    syn0 += np.dot(l0.T,l1_delta) # (3x4) dot (4x1) = (3x1)

print("Ouput After Training:", l1)


"""

Reference:
    
X 	        Input dataset matrix where each row is a training example

y 	        Output dataset matrix where each row is a training example

l0 	        First Layer of the Network, specified by the input data

l1 	        Second Layer of the Network, otherwise known as the hidden layer

l1_error    Relative error after each prediction compare to output

l1_delta    Magnitude to the true error each time

syn0        First layer of weights, Synapse 0, connecting l0 to l1.

*	        Elementwise multiplication, so two vectors of equal size are multiplying corresponding values 1-to-1 to generate a final vector of identical size.

-	        Elementwise subtraction, so two vectors of equal size are subtracting corresponding values 1-to-1 to generate a final vector of identical size.

x.dot(y) 	If x and y are vectors, this is a dot product. If both are matrices, it's a matrix-matrix multiplication. If only one is a matrix, then it's vector matrix multiplication.

https://iamtrask.github.io/2015/07/12/basic-python-network/

"""