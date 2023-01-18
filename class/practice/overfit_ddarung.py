import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

path = './_data/ddarung/'
train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv = pd.read_csv(path+'test.csv',index_col=0)
submission = pd.read_csv(path+'submission.csv',index_col=0)

train_csv = train_csv.dropna()
x = train_csv.drop(['count'],axis =1)
y = train_csv['count']

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=3432)

# print(x_train.shape)
# print(x_test.shape)


model = Sequential()
model.add(Dense(100,input_dim=9 ))
model.add(Dense(200,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(1))

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

