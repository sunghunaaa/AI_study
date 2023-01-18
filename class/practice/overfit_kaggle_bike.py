import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

path = './_data/bike/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
sample= pd.read_csv(path+'sampleSubmission.csv', index_col=0)

x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.85,shuffle=True,random_state=9)

model = Sequential()
model.add(Dense(16,input_dim = 8, activation='linear'))
model.add(Dense(32, activation='linear'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))

#3.compile, fit

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train,epochs=10, batch_size=1 
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
print(hist.history['loss']) 
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

