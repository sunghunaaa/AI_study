import numpy as np
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=321)
scaler =MinMaxScaler()
x_train =scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

input1 = Input(shape=(13,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(5, activation='linear')(dense1)
output1 = Dense(1,activation='linear')(dense2)
model = Model(inputs=input1, outputs=output1)
model.summary
path = './_data/'
#path = 'C:/study/_save/'  절대 경로
model.save(path + 'keras29_1_save_model.h5')
#model.save('./_data/keras29_1_save_model.h5') 위 줄과 동일
