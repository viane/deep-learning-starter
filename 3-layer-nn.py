"""
  Inputs	   |  Output
0	0	1  |	    0
0	1	1  |	    1
1	0	1  |	    1
1	1	1  |    0

It appears to be completely unrelated to column three, 
which is always 1. However, columns 1 and 2 give more clarity. 
If either column 1 or 2 are a 1 (but not both!) then the output is a 1. 
This is our pattern.

3 Layer Neural Network
"""



import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2
    
    if (j% 10000) == 0:
        print("Error:", str(np.mean(np.abs(l2_error))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)



"""

Reference:
    
X	Input dataset matrix where each row is a training example

y	Output dataset matrix where each row is a training example

l0	First Layer of the Network, specified by the input data

l1	Second Layer of the Network, otherwise known as the hidden layer

l2	Final Layer of the Network, which is our hypothesis, and should approximate the correct answer as we train.

syn0	First layer of weights, Synapse 0, connecting l0 to l1.

syn1	Second layer of weights, Synapse 1 connecting l1 to l2.

l2_error	This is the amount that the neural network "missed".

l2_delta	This is the error of the network scaled by the confidence. It's almost identical to the error except that very confident errors are muted.

l1_error	Weighting l2_delta by the weights in syn1, we can calculate the error in the middle/hidden layer.

l1_delta	This is the l1 error of the network scaled by the confidence. Again, it's almost identical to the l1_error except that confident errors are muted.

*	        Elementwise multiplication, so two vectors of equal size are multiplying corresponding values 1-to-1 to generate a final vector of identical size.

-	        Elementwise subtraction, so two vectors of equal size are subtracting corresponding values 1-to-1 to generate a final vector of identical size.

x.dot(y) 	If x and y are vectors, this is a dot product. If both are matrices, it's a matrix-matrix multiplication. If only one is a matrix, then it's vector matrix multiplication.

https://iamtrask.github.io/2015/07/12/basic-python-network/

"""