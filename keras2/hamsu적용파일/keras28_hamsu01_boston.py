import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import tensorflow as tf
import numpy as np

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=321)

#2. model(순차형)
model= Sequential()
model.add(Dense(10,activation='relu',input_dim=13))
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='linear'))
model.add(Dense(1,activation='linear'))

#2. model(함수형)

