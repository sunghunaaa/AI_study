import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
##############################################################
from tensorflow.python.keras.models import load_model
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
##############################################################
model = load_model(path + 'MCP/keras30_ModelCheckPoint1.hdf5')
#중간 모델구성, 컴파일, 훈련코드 생략, 위의 load_model 사용 
##############################################################
#4. evaluate,predict


