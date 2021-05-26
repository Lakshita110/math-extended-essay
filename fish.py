# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from mxnet import nd, autograd, gluon


# %%
df = pd.read_csv("fish.csv")
fishes = df[['Weight','Height']]


# %%
training = fishes.sample(frac = 0.5)
testing = fishes.drop(training.index)
x_train = training["Weight"].to_list() 
y_train = training["Height"].to_list()


# %%
plt.scatter(x_train,y_train)
plt.show()


# %%
num_inputs = 1
num_outputs = 1


# %%
w = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs)
print(w,b)


# %%



