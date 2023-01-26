import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
#1. data
dataset = np.array([1,2,3,4,5,6,7,8,9,10])
x= np.array([[1,2,3],   
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],
             [8,9,10],
             [9,10,11],
             [10,11,12],
             [20,30,40],
             [30,40,50],
             [40,50,60]])
y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x = x.reshape(13,3,1)

#2. model
model = Sequential()
"""
model.add(LSTM(units=64,input_shape=(3,1), ret)) # input (13,3,1) output (None, 64)  
model.add(LSTM(32)) #input이 2차원이라 error // ValueError: Input 0 of layer "lstm_1" is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 64)
"""
model.add(LSTM(units=64,input_shape=(3,1), return_sequences=True)) # output  (None, 3, 64)  ## return_sequences=False가 default
model.add(LSTM(32)) # input (None, 3, 64) output (None, 32)
#첫번째 LSTM output을 3차원을 해줘서 LSTM 두번 쓸 수 있던 거임

model.add(Dense(64, activation='relu')) 
model.add(Dense(1))
model.summary()


















#3. compile, fit
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1, verbose=2)

#4. predict
loss = model.evaluate(x,y)
print(loss)
y_pred = np.array([50,60,70]).reshape(1,3,1)
result = model.predict(y_pred)
print(result)






"""
0.08286876231431961
[[74.26124]]
"""
