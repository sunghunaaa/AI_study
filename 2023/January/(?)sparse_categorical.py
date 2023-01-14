### sparse_categorical은 1부터 시작하는 data에는 적용이 안 됨// 0부터 시작하는 data에만 적용
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
# print(x.shape, y.shape) #(581012, 54) (581012,)
# print(np.unique(y,return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
# print("return : ", np.unique(y)) #return :  [1 2 3 4 5 6 7]
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=321,stratify=y)
#2. model
model = Sequential()
model.add(Dense(10,activation='relu',input_dim=54))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(10,activation='linear'))
model.add(Dense(7,activation='softmax'))
#3. compile,fit
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2,batch_size=32,validation_split=0.3,verbose=1)
#4. evaluate,predict
loss, accuracy = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
"""
print('1 :',y_predict)
1 : [[2.2521669e-10 4.2932138e-01 4.9499109e-01 ... 9.9863512e-03
  9.3242265e-03 8.4898053e-03]
 [2.2521669e-10 4.2932138e-01 4.9499109e-01 ... 9.9863512e-03
  9.3242265e-03 8.4898053e-03]
 [2.2521669e-10 4.2932138e-01 4.9499109e-01 ... 9.9863512e-03
  9.3242265e-03 8.4898053e-03]
 ...
 [2.2521669e-10 4.2932138e-01 4.9499109e-01 ... 9.9863512e-03
  9.3242265e-03 8.4898053e-03]
 [2.2521669e-10 4.2932138e-01 4.9499109e-01 ... 9.9863512e-03
  9.3242265e-03 8.4898053e-03]
 [2.0282915e-08 1.8960194e-01 7.8319669e-01 ... 4.9086367e-03
  1.2530050e-02 3.5136156e-03]]
print(y_test)
2 : [2 7 1 ... 1 7 2]
print(y_predict.shape)
(116203, 7)
"""
y_predict = np.argmax(y_predict, axis =1)
print('1:',y_predict) #1: [2 2 2 ... 2 2 2]
print('2:',y_predict.shape)# 2: (116203,)
"""
print('3:',y_test) #3: [2 7 1 ... 1 7 2]
print('4:',y_test.shape) #4: (116203,)
y_test 자체가 0차원이라 np.argmax 안 됨
y_test = np.argmax(y_test, axis=1)
print('5:',y_test)
print('6:',y_test.shape)
"""
acc = accuracy_score(y_test,y_predict)
print(acc)
