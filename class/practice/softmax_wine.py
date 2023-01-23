import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

#1. data
datasets = load_wine()
x = datasets.data
y = datasets.target

#print(x.shape, y.shape) #(178,13) (178,)
#print(y)  # 0,1,2로 이루어졌는 것을 확인
#print(np.unique(y)) #[0 1 2]  => *****y는 0과 1과 2만 있다는 것을 알 수 있음 *****
print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48])) -> 0이 59개 , 1이 71개, 2가 48개 있음을 의미
y= to_categorical(y)
print(y.shape)#(178, 3)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=321, stratify=y)
print(y_train.shape)#(124, 3)
print(x_train.shape)#(124, 13)





model= Sequential()
model.add(Dense(10,activation='relu',input_dim=13))
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='linear'))
model.add(Dense(3,activation='softmax'))


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=145,batch_size=1,validation_split=0.2,verbose=1)
loss, accuracy = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
print(y_predict)
y_predict = np.argmax(y_predict,axis =1)
print('===================================================')
print(y_test)
y_test = np.argmax(y_test,axis =1)
print(y_test)
print('===================================================')
acc = accuracy_score(y_test,y_predict)
print(acc)
