import numpy as np
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
###########################################################
from tensorflow.python.keras.layers import Dropout
###########################################################
path = './_save/'
#1.data
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test=train_test_split (x,y,test_size=0.3,shuffle=True,random_state=321)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#2.model
input1 = Input(shape=(13,))
dense1 = Dense(10,activation='relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(10,activation='linear')(drop1)
drop2 = Dropout(0.5)(dense2)
output1 = Dense(1,activation='linear')(drop2)
model = Model(inputs=input1, outputs=output1)
model.summary()
#3. compile,fit
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2,verbose=1)
#4. evaluate,predict
mse, mae = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

