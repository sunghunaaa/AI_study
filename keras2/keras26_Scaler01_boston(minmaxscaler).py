import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target


print(x) 
print(type(x))  #<class 'numpy.ndarray'> sklearn에 x는 numpy이다.
scaler = MinMaxScaler()
scaler.fit(x) # fit은 범위만큼 scaler의 가중치를 생성해준 것이고
x = scaler.transform(x) # 실제로 x를 바꿔주는 과정
print(('최솟값 : '),np.min(x))
      
print(('최댓값 : '),np.max(x))
print(x.shape)
print(y.shape)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=321)
model= Sequential()
model.add(Dense(10,activation='relu',input_dim=13))
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='linear'))
model.add(Dense(1,activation='linear'))

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mae'])
model.fit(x_train,y_train,epochs=1000,batch_size=32,validation_split=0.2,verbose=1)
loss, mae = model.evaluate(x_test, y_test)
print('lmse : ', loss)
print('lmae : ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   #->mean_squared_error(y_test, y_predict) = 'mae", sqrt=> 루트

print("RMSE : ", RMSE(y_test,y_predict))
  
r2= r2_score(y_test,y_predict)
print("R2 : ", r2)


# 성능 비교 
# 전RMSE :  
# 4.491956907547301
#R2 :  0.7483300778194151

# 후minmaxscaler
#RMSE :  4.027463954745218
# R2 :  0.7976871459998462

# 후standardscaler
# RMSE :  4.847114614866514
# R2 :  0.7069601181605337
