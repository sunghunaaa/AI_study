# [실습]
# 1.train 0.7이상
# 2.R2 : 0.8이상 / RMSE 사용
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
# print(x)
# print(x.shape)  #(506, 13)
# print(y)
# print(y.shape)  #(506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,shuffle=True,random_state=79)
# print(x_test.shape)  #(,)
# print(x_train.shape)  #(,)
# print(y_train.shape)  #(,)
# print(y_test.shape)  #(,)


# print(dataset.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

# print(dataset.DESCR)

#2 model
model = Sequential()
model.add(Dense(500, input_dim=13))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(1))

#3 compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size=3)


loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  
print("RMSE : ", RMSE(y_test,y_predict))
  
r2= r2_score(y_test,y_predict)
print("R2 : ", r2)


# loss :  26.756160736083984
# RMSE :  5.172636375780111
# R2 :  0.7693810548225117


#Dense epochs 변경

# loss :  7.133237361907959
# RMSE :  2.670811970213496
# R2 :  0.8066881209261916


