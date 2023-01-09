import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

path = './_data/ddarung/'
train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv = pd.read_csv(path+'test.csv',index_col=0)
submission = pd.read_csv(path+'submission.csv',index_col=0)

train_csv = train_csv.dropna()
x = train_csv.drop(['count'],axis =1)
y = train_csv['count']

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=3432)

model = Sequential()
model.add(Dense(18,input_dim=9 ))
model.add(Dense(36,activation='relu'))
model.add(Dense(36,activation='relu'))
model.add(Dense(18,activation='relu'))
model.add(Dense(1))

#3.compile, fit

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train,epochs=800, batch_size=3   # hist = history의 준말 
          ,validation_split= 0.2, 
          verbose = 1
          )

#4. evaluate,predict
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test) 

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('loss : ', loss)
print('RMSE : ', RMSE)

y_submission = model.predict(test_csv)
submission['count'] = y_submission
submission.to_csv(path+'submission0109#1.csv')






