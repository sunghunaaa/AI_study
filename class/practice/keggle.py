# https://www.kaggle.com/competitions/bike-sharing-demand

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

path = './_data/bike/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
sample= pd.read_csv(path+'sampleSubmission.csv', index_col=0)

x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']


# print(x.shape)  10886,8
# print(y.shape)  10886


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=7)



model = Sequential()
model.add(Dense(16,input_dim = 8,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=4000,batch_size=23,
          validation_data=(x_test,y_test)
          )

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print(rmse)
y_submit = model.predict(test_csv)

sample['count'] = y_submit

sample.to_csv(path+'bike_01064.csv')


# 154.3988008825224 
# 148.76319199272876
# 179.07404236122747
# 150.54777045734514
# 162.06856600823582
# 148.32057822676356  01063
# 147.32473882189572
