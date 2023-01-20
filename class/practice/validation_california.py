#[실습]
#R2 0.55~0.6 이상

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing


#1. data
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size= 0.8,random_state=709)

model = Sequential()
model.add(Dense(26,input_dim=8))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(26,activation='relu'))
model.add(Dense(13,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=3,
          validation_split=0.25)

loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

y_predict = model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print('rmse : ', RMSE(y_test,y_predict))

r2 =r2_score(y_test,y_predict)
print('r2 : ', r2)
