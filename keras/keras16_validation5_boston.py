# [실습]
# 1.train 0.7이상
# 2.R2 : 0.8이상 / RMSE 사용

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston

#1. data
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size= 0.84,random_state=31)

model = Sequential()
model.add(Dense(100,input_dim=13))
model.add(Dense(200,activation='relu'))
model.add(Dense(300))
model.add(Dense(300,activation='relu'))
model.add(Dense(300))
model.add(Dense(300,activation='relu'))
model.add(Dense(300))
model.add(Dense(300,activation='relu'))
model.add(Dense(300))
model.add(Dense(300,activation='relu'))
model.add(Dense(200))
model.add(Dense(100,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=500,batch_size=2,
          validation_split=0.30)

loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

y_predict = model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print('rmse : ', RMSE(y_test,y_predict))

r2 =r2_score(y_test,y_predict)
print('r2 : ', r2)

