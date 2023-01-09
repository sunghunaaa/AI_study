import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

path = './_data/ddarung/' 
train_csv = pd.read_csv(path+'train.csv', index_col=0) 
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

train_csv = train_csv.dropna() 

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=9)

model = Sequential()
model.add(Dense(9,input_dim=9))
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(1))

import time

model.compile(loss = 'mse' ,optimizer = 'adam')
start = time.time()
model.fit(x_train,y_train,epochs=10000,batch_size=21)
end = time.time()


loss = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
print("걸린 시간 : " , end - start)
#tf27 cpu = 71초
#tf274 gpu =

y_submit = model.predict(test_csv)

submission['count'] = y_submit
print(submission)

submission.to_csv(path+ 'submission_0105.csv')


