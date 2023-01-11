import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import tensorflow as tf


#1. data
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape,y.shape) #(581012, 54) (581012,)
print(np.unique(y,return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510 ]



###################################1.케라스 투카테고리컬#################################################################
# y= pd.get_dummies(y,ignore_index=True)
from tensorflow.keras.utils import to_categorical
y= to_categorical(y)
print(y.shape) #(581012,8)
print(type(y)) #<class 'numpy.ndarray'>
print(np.unique(y[:,0], return_counts=True))
#(array([0.], dtype=float32), array([581012], dtype=int64))
print(np.unique(y[:,1], return_counts=True))
#(array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))
y = np.delete(y,0,axis=1)
print(y.shape) #(581012, 7)
print(y[:10])
print(np.unique(y[:,0], return_counts=True)) #(array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))
#=======================================================================================================================
    
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=321, stratify=y)

model= Sequential()
model.add(Dense(10,activation='relu',input_dim=54))
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='linear'))
model.add(Dense(7,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2,batch_size=32,validation_split=0.2,verbose=1)
loss, accuracy = model.evaluate(x_test, y_test)
print('y_test(1)',y_test.shape)

y_predict = model.predict(x_test)
print('y_predict(1)',y_predict.shape)
y_predict = np.argmax(y_predict,axis =1)
print('y_predict(2)',y_predict.shape)
y_test = np.argmax(y_test,axis =1)
print(y_test.shape)  #자료형 확인하래 tensorflow와 numpy 자료형이 다름
acc = accuracy_score(y_test,y_predict)
print(acc)
















