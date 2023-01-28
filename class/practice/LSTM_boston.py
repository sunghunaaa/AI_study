import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error,r2_score
import tensorflow as tf

#1.data
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape)  #(506, 13)
print(y.shape)  #(506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,shuffle=True,random_state=79)
print(x_test.shape)  #(,) (51, 13)
print(x_train.shape)  #(,) (455, 13)
print(y_train.shape)  #(,) (455,)
print(y_test.shape)  #(,) (51,)


# print(dataset.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']
# print(dataset.DESCR)

x_test = x_test.reshape(51,13,1)
x_train = x_train.reshape(455,13,1)



from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(LSTM(64, input_shape=(13,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3.compile, fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=10,batch_size=1,validation_split=0.1)

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
RMSE :  4.24607007698156
R2 :  0.5114081377638757
"""
