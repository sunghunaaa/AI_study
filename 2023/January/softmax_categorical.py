import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf

#1. data
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(x.shape) #(581012, 54)
# print(y.shape) #(581012,)
# print(np.unique(y, return_counts=True))  #(array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

# keras to_cateforical
from keras.utils import to_categorical
y = to_categorical(y)
#print(y.shape)  #(581012, 8)
#print(type(y)) #<class 'numpy.ndarray'>
#print(np.unique(y[:,1],return_counts=True)) #(array([0., 1.], dtype=float32), array([369172, 211840]))
#print(np.unique(y[:,0],return_counts=True)) #(array([0.], dtype=float32), array([581012]))
#print(np.unique(y,return_counts=True)) #(array([0., 1.], dtype=float32), array([4067084,  581012]))  
y = np.delete(y,0,axis=1) #열 삭제, 의미없이 생긴 0(1열)삭제하는 것
#print(y.shape) #(581012, 7) 잘 잘린 것을 확인한다.
#print(np.unique(y[:,0],return_counts=True))#(array([0., 1.], dtype=float32), array([369172, 211840])) 1(1열)잘 살아남은지 확인
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,shuffle=True,random_state=321,stratify=y)
"""
stratify: stratify 파라미터는 분류 문제를 다룰 때 매우 중요하게 활용되는 파라미터 값 입니다. stratify 값으로는 target 값을 지정해주면 됩니다.
stratify값을 target 값으로 지정해주면 target의 class 비율을 유지 한 채로 데이터 셋을 split 하게 됩니다. 만약 이 옵션을 지정해주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있습니다. 
*0과 1뿐이 데이터에서 한 쪽으로 치우쳐진 데이터를 비율을 유치 한 채 split하는데 아주 중요한 역할을 하는 파라미터 값임.*
"""
#2. model
model = Sequential()
model.add(Dense(10,activation='relu',input_dim=54))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(10,activation='linear'))
model.add(Dense(10,activation='relu'))
model.add(Dense(7,activation='softmax'))
#3. compile
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=1,batch_size=32,validation_split=0.2,verbose=1)
#4. evaluate, predict
loss, acccuracy = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
#print(y_predict)
"""
[[7.1537614e-01 8.7372042e-02 2.2298980e-07 ... 1.8856749e-04
  1.0908473e-05 1.9705214e-01]
 [5.0245398e-01 4.4854546e-01 5.1475689e-04 ... 1.8826578e-02
  2.5991946e-03 2.7059985e-02]
 [2.7668694e-01 6.2750399e-01 6.6788979e-03 ... 6.8096302e-02
  1.5489644e-02 5.5435207e-03]
 ...
 [6.5834248e-01 2.6492831e-01 3.8754392e-06 ... 7.7515084e-04
  6.4883599e-05 7.5885251e-02]
 [6.4377922e-01 8.3263636e-02 1.2641469e-06 ... 4.3038843e-04
  1.8510988e-05 2.7250695e-01]
 [2.6399037e-01 7.2426897e-01 2.6240729e-04 ... 4.2077075e-03
  5.0632923e-04 6.7641917e-03]]
 이런 굉장한 짤짤이 값으로 나옴, 한 행의 총합이 확률 100%에 기반된 1임 따라서 행 중 가장 높은 값이 1이 되고 나머지는 0이 되는 작업을 해주어야 함
  """
"""
y_predict = np.argmax(y_predict,axis=1)
print(y_predict)
acc = accuracy_score(y_test,y_predict)
print(acc)
numpy된 y_predict와 categorical된 y_test는 서로의 모양이 다름. 따라서 y_test도 argmax형태로 바꿔줘야 함.
  y_predict [1 1 1 ... 1 1 1]
  y_test
    [[1. 0. 0. ... 0. 0. 0.]
    [0. 1. 0. ... 0. 0. 0.]
    [0. 1. 0. ... 0. 0. 0.]
     ...
    [1. 0. 0. ... 0. 0. 0.]
    [0. 0. 0. ... 0. 0. 1.]
    [0. 1. 0. ... 0. 0. 0.]]
"""

y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_predict)
print(acc)
#0.5324490545254268
