import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pandas as pd
import tensorflow as tf
#1. data
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(x.shape,y.shape) #(581012, 54) (581012,)
# print(np.unique(y,return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
# print(y) #[5 5 2 ... 3 3 3]

# one hot encoding 할 때 y 값은 1차원이 아닌 2차원이여야 가능하다.
y = y.reshape(581012,1)
# print(y)
"""[[5]
 [5]
 [2]
 ...
 [3]
 [3]
 [3]]
 """
# sklearn one hot encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder() 
 
 ////
