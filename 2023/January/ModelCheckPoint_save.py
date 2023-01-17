import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
##############################################################
from tensorflow.python.keras.callbacks import ModelCheckpoint
##############################################################
path = './_save/'
#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=321)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#2. model
input1 = Input(shape=(13,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(10, activation='relu')(dense1)
output1 = Dense(1, activation='linear')(dense2)
model = Model(inputs=input1, outputs=output1)
model.summary()
#3. compile,fit
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
# 모델을 더이상 학습을 못 할 경우(loss, metrics등의 개선이 없을 경우), 학습 도중 미리 학습을 종료시키는 콜백함수 earlystopping
es = EarlyStopping(monitor='val_loss',mode='min',patience=1,restore_best_weights=True,verbose=1) 
##############################################################
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only=True, filepath= path +'MCP/keras30_ModelCheckPoint1.hdf5')
# save_best_only : 가중치 가장 좋은 지점 저장
##############################################################
model.fit(x_train,y_train,epochs=2,batch_size=32,validation_split=0.3,verbose=1,callbacks=[es,mcp])
model.save(path + 'keras30_1_save_model.h5') # 모델 저장 (가중치 포함 x)

"""
!!저장되는 파일 명을 시간별로 찍는 방법.
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
print(type(date)) <class 'str>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #0037-0.0048.hdf 

mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only=True, filepath= filepath+'k30_'+date+'_'+filename)

# save_best_only : 가중치 가장 좋은 지점 저장
"""

#4. evaluate,predict


