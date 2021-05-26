#%% 
import numpy as np 
import random as rd
from fish import x_train, y_train 
#%%
def loss(o, y):
    # mean squared error loss
    return (y-o)**2 

def cost(x,y,w,b):
    total_cost = 0.0 
    # sum of all costs over the training data 
    for i in range(len(x)):
        output = x[i]*w + b
        total_cost += loss(output,y[i])

    return total_cost

def update_weights(x,y,w,b,lr):
    weight_deriv = 0
    bias_deriv = 0 
    fishes = len(x)

    for i in range(fishes):
        output = x[i]*w + b
        # calculate partial derivative of cost function
        # -2x(y-(wx+b))
        weight_deriv += -2*x[i]*(y[i] - output)

        # -2(y-(wx+b))
        bias_deriv += -2*(y[i] - output)

    w -= (weight_deriv / fishes) * lr
    b -= (bias_deriv / fishes) * lr 

    return w, b

def train(x,y,w,b,lr,iters):
    cost_history=[]

    for i in range(iters):
        w,b = update_weights(x,y,w,b,lr)

        c = cost(x,y,w,b)
        cost_history.append(c)

        # Log Progress
        if i % 10 == 0:
            print ("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, w, b, c))

    return w, b, cost_history
#%%
print(train(x_train, y_train, 0.1, 1, 0.1, 1000))
