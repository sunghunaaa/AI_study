#RNN의 일종
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM


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
"""
simplernn
# model.add(SimpleRNN(units=64, input_shape=(3,1))) #(N, 3, 1) -> (batch[행, 데이터 개수], timesteps[여기선 3개씩 자름의 3개부분을 의미,y값이 없음], feature[1개씩 훈련해서]) 
# 1. model.add(SimpleRNN(units=64, input_shape=(3,1)))
# 2. model.add(SimpleRNN(units=64,input_length=3 ,input_dim=1 ))
# 3. model.add(SimpleRNN(units=64,input_dim=1 ,input_length=3 ))
# 1,2,3 동일
# 치명적인 단점 : 큰 데이터에서보면 timesteps가 크다. timesteps가 크면 실제 연산을 했을 때 앞데이터가 의미가 없어져서
"""
#model.add(SimpleRNN(units=10, input_shape=(3,1))) # simplernn parameter 120
model.add(LSTM(units=10,input_shape=(3,1))) # LSTM parameter 480  // https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.summary()
