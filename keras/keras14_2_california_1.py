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

model = Sequential()
model.add(Dense(500,input_dim=8))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100,batch_size=15)

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  
print("RMSE : ", RMSE(y_test,y_predict))
  
r2= r2_score(y_test,y_predict)
print("R2 : ", r2)

# loss :  0.5838719010353088
# RMSE :  0.9634254856510992
# R2 :  0.3266177950223095
# # batch, loss, epchs
# loss :  0.6871798038482666
# RMSE :  0.8289630413568314
# R2 :  0.5014649343722388
# #epochs50 ->100
# loss :  0.6559832692146301
# RMSE :  0.8099279003955954
# R2 :  0.5240973821175248

#size 0.9 -> 0.95


