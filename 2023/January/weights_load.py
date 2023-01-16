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
#3. compile,fit
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
earlystopping = EarlyStopping(monitor='val_loss',patience=3,mode='min',restore_best_weights=True)
"""
load_weights : 가중치만 저장되어, 모델에서 사용 못 함
위에서 모델이 정의되어야 사용 가능
순수하게 가중치만 저장되어 있어, 컴파일, 훈련을 사용 안 하고 실행하면 컴파일 에러 발생 
결국에는 가중치만 저장되어 있어 컴파일에 대한 명시가 없기 떄문에 직접 컴파일을 사용하여 실행
fit 훈련 단계 필요없고 compile로 loss함수 optimizer만 정해주면 됨
"""
model.load_weights(path + 'keras29_5_save_weights2.h5')
#4. evaluate,predict

