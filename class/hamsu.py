# 모델에는 sequential 모델, 함수형 모델 등이 있음.
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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. model(순차형)
# model= Sequential()
# model.add(Dense(10,activation='relu',input_dim=13))
# model.add(Dense(20,activation='sigmoid'))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(20,activation='linear'))
# model.add(Dense(1,activation='linear'))
# model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 10)                140

#  dense_1 (Dense)             (None, 20)                220

#  dense_2 (Dense)             (None, 20)                420

#  dense_3 (Dense)             (None, 20)                420

#  dense_4 (Dense)             (None, 1)                 21

# =================================================================
# Total params: 1,221
# Trainable params: 1,221
# Non-trainable params: 0


# ####################################함수형 모델############################################
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense

#2. model(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(20, activation='sigmoid')(dense1)
dense3 = Dense(20, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()


#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 13)]              0

#  dense (Dense)               (None, 10)                140

#  dense_1 (Dense)             (None, 20)                220

#  dense_2 (Dense)             (None, 20)                420

#  dense_3 (Dense)             (None, 20)                420

#  dense_4 (Dense)             (None, 1)                 21

# =================================================================
# Total params: 1,221
# Trainable params: 1,221
# Non-trainable params: 0
