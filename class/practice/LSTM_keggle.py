# https://www.kaggle.com/competitions/bike-sharing-demand

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


path = './_data/bike/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
sample= pd.read_csv(path+'sampleSubmission.csv', index_col=0)

x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']


# print(x.shape)  10886,8
# print(y.shape)  10886
x=x.values
y=y.values

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=7)

print(x_test.shape)  #(,) (1032, 8)
print(x_train.shape)  #(,) (19608, 8)
print(y_train.shape)  #(,)(19608,)
print(y_test.shape)  #(,) (1032,)



x_test = x_test.reshape(2178, 8,1)
x_train = x_train.reshape(8708, 8,1)

from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(LSTM(64, input_shape=(8,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3.compile, fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=2,batch_size=32,validation_split=0.1)

#4. evaluate
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print(y_predict)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  
print("RMSE : ", RMSE(y_test,y_predict))
  
r2= r2_score(y_test,y_predict)
print("R2 : ", r2)

# RMSE :  172.10637798555487
# R2 :  0.07612197771513074
