import numpy as np
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
#1.data
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=321)
scaler =MinMaxScaler()
x_train =scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#2.model
input1 = Input(shape=(13,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(5, activation='linear')(dense1)
output1 = Dense(1,activation='linear')(dense2)
model = Model(inputs=input1, outputs=output1)
model.summary

########################################################
path = './_data/'
#path = 'C:/study/_save/'  절대 경로
model.save(path + 'keras29_1_save_model.h5')
#model.save('./_data/keras29_1_save_model.h5') 위 줄과 동일
########################################################

#3.compile,fit
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
earlystopping = EarlyStopping(monitor='val_loss',mode='min',patience=4,restore_best_weights=True)
model.fit(x_train,y_train,epochs=2,batch_size=32,validation_split=0.3,callbacks=[earlystopping],verbose=1)

########################################################
path = './_data/'
#path = 'C:/study/_save/'  절대 경로
model.save(path + 'keras29_1_save_model_compile.h5')
#model.save('./_data/keras29_1_save_model_compile.h5') 위 줄과 동일
########################################################

"""
model save 위치에 따라 모델만 저장할 수 있고/ complie,fit 한 뒤에 save를 하면 model,compile,fit 모두를 저장할 수 있다.
"""
