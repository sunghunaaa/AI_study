from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense,Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
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
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(5, activation='linear')(dense1)
output1 = Dense(1,activation='linear')(dense2)
model = Model(inputs=input1, outputs=output1)
model.save_weights(path + 'keras29_5_save_weights1_1.h5')  #초기 weight 값 저장됨, 직접 실행시켜봄
#3. compile,fit
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
earlystopping = EarlyStopping(monitor='val_loss',patience=3,mode='min',restore_best_weights=True)
model.fit(x_train,y_train,epochs=2,batch_size=32,validation_split=0.2,callbacks=[earlystopping],verbose=1)
model.save_weights(path + 'keras29_5_save_weights1_2.h5')  #complile, fit 후에 save weight하면 좋은 weight값 저장됨
#4. evaluate,predict


