import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


#1. data
dataset = np.array([1,2,3,4,5,6,7,8,9,10])
x= np.array([[1,2,3],   
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9]])
y= np.array([4,5,6,7,8,9,10])


x = x.reshape(7,3,1)

#2. model
model = Sequential()
model.add(SimpleRNN(units = 256, input_shape=(3,1))) #(N, 3, 1) -> (batch[행, 데이터 개수], timesteps[y값이 없음], feature[1개씩 훈련해서]) 
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn (SimpleRNN)      (None, 256)               66048  = 256^2 +256*1 + 256

 dense (Dense)               (None, 64)                16448

 dense_1 (Dense)             (None, 1)                 65

=================================================================
Total params: 82,561
Trainable params: 82,561
Non-trainable params: 0
_________________________________________________________________
"""