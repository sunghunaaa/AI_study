import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pandas as pd
import tensorflow as tf
#1. data
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(x.shape,y.shape) #(581012, 54) (581012,)
# print(np.unique(y,return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
# print(y) #[5 5 2 ... 3 3 3]

# one hot encoding 할 때 y 값은 1차원이 아닌 2차원이여야 가능하다.
y = y.reshape(581012,1)
# print(y)
"""[[5]
 [5]
 [2]
 ...
 [3]
 [3]
 [3]]
 """
############## sklearn one hot encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y)
# y = ohe.fit_transform(y) 위 두 줄을 한줄로 표현한 것.
#print(y)
"""
(0, 4)        1.0
(1, 4)        1.0
(2, 1)        1.0
:     :
(581009, 2)   1.0
(581010, 2)   1.0
(581011, 2)   1.0
Scipy 형태라 numpy와 오류남 따라서 numpy형태로 변형이 필요함.
"""
y = y.toarray()
#print(y.shape)#(581012, 7)
#print(y)
"""
[[0. 0. 0. ... 1. 0. 0.]
 [0. 0. 0. ... 1. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]
 ...
 [0. 0. 1. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]]
"""
x_train,y_train,x_test,y_test = train_test_split(x ,y ,test_size=0.3 ,shuffle=True ,random_state=321 ,stratify=y)
#2. model
model = Sequential()
model.add(Dense(10,activation='relu',input_dim=54))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(10,activation='linear'))
model.add(Dense(7,activation='softmax'))
#3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam',metics=['accuracy'])
model.fit(x_train,y_train,epochs=2,batch_size=32,validation_split=0.2,verbose=1)
#4. evaluate, predict
loss, accuracy = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_predict)
print(acc)
