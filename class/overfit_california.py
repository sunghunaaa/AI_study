import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import fetch_california_housing


#1. data
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size= 0.8,random_state=709)

model = Sequential()
model.add(Dense(26,input_dim=13))
model.add(Dense(30 ,activation='relu'))
model.add(Dense(30 ,activation='relu'))
model.add(Dense(30 ,activation='relu'))
model.add(Dense(26 ,activation='relu'))
model.add(Dense(13 ,activation='relu'))
model.add(Dense(1))

#3.compile, fit

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train,epochs=1000, batch_size=1
          ,validation_split= 0.2, 
          verbose = 1
          )

#4. evaluate,predict
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)
print("====================================")
print( hist)
print("====================================")
print(hist.history)
print("====================================")
print(hist.history['loss'] 
print("====================================")
print(hist.history['val_loss'])
print("====================================")

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],c = 'red', marker = '.', label = 'loss')
plt.plot(hist.history['val_loss'], c='blue', marker = ' .',label = 'val_loss' )
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('boston loss')
plt.legend(loc='upper right')
plt.show()

