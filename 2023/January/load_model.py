import numpy as np
from tensorflow.python.keras.models import Sequential, Model, load_model ## load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=321)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#2. model
path = './_data/'
model = load_model(path + 'keras29_1_save_model.h5') # 저장된 모델 불러오기
model.summary() 
# 모델 구성 같음
# compile, fit 전에 저장하면 모델 구성만 불러옴
#3. compile,fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.python.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', mode='min',patience=1,restore_best_weights=True,verbose=1)
model.fit(x_train,y_train,epochs=1,batch_size=32,verbose=1,validation=0.2,callbacks=[earlystopping])
#4. evaluate,predict
mse, mae = model.evaluate(x_train,y_train)
y_predict = model.predict(x_test)
r2_score = r2_score(y_test,y_predict)
