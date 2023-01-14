import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#1. data
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(x.shape,y.shape) #(581012, 54) (581012,)
# print(np.unique(y,return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
# print(y) #[5 5 2 ... 3 3 3]
#############pandas
import pandas as pd
y = pd.get_dummies(y)
#print(y[:10])
"""
   1  2  3  4  5  6  7
0  0  0  0  0  1  0  0
1  0  0  0  0  1  0  0
2  0  1  0  0  0  0  0
3  0  1  0  0  0  0  0
4  0  0  0  0  1  0  0
5  0  1  0  0  0  0  0
6  0  0  0  0  1  0  0
7  0  0  0  0  1  0  0
8  0  0  0  0  1  0  0
9  0  0  0  0  1  0  0
"""
#print(y.shape) #(581012, 7)
#print(type(y)) #<class 'pandas.core.frame.DataFrame'>
"""
ValueError: Shaep of passed value is (174304, 1), indices imply (174304, 7)
현재 y의 데이터의 형태는 pandas임 따라서 np.argmax를 받아드릴 수 없어 생기는 error임
* argmax tensorflow는 pandas나 numpy 다 받아드릴 수 있음
"""
"""
해결방법
1. values
y = y.values # y data type pandas 에서 numpy로 바꿈
print(type(y)) # <class 'numpy.ndarray'>

2. to_numpy
y = y.to_numpy()
print(type(y)) # <class 'numpy.ndarray'>
"""
y = y.to_numpy()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=321,stratify=y)

#2. model
model = Sequential()
model.add(Dense(10,activation='relu',input_dim=54))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(10,activation='linear'))
model.add(Dense(7,activation='softmax'))
#3. compile,fit
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2,batch_size=32,validation_split=0.2,verbose=1)
#4. evaluate, predict
loss, accuracy = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_predict)
print(acc)
