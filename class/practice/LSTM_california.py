#[실습]
#R2 0.55~0.6 이상
import numpy as np
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
#print(datasets.feature_names)
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
#print(datasets.DESCR) 
#         - MedInc        median income in block group
#         - HouseAge      median house age in block group
#         - AveRooms      average number of rooms per household
#         - AveBedrms     average number of bedrooms per household
#         - Population    block group population
#         - AveOccup      average number of household members
#         - Latitude      block group latitude
#         - Longitude     block group longitude
# print(x)
# print(x.shape)  # (20640, 8)
# print(y)
# print(y.shape)  # (20640,)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.95,random_state=31)
print(x_test.shape)  #(,) (1032, 8)
print(x_train.shape)  #(,) (19608, 8)
print(y_train.shape)  #(,)(19608,)
print(y_test.shape)  #(,) (1032,)

x_test = x_test.reshape(1032, 8,1)
x_train = x_train.reshape(19608, 8,1)

from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(LSTM(64, input_shape=(8,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3.compile, fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=2,batch_size=1,validation_split=0.1)

#4. evaluate
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print(y_predict)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  
print("RMSE : ", RMSE(y_test,y_predict))
  
r2= r2_score(y_test,y_predict)
print("R2 : ", r2)
"""
RMSE :  0.6775085963368801
R2 :  0.6653140489365597
"""
