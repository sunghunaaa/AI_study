import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

#1. data
datasets = load_wine()
x = datasets.data
y = datasets.target

print(np.unique(y, return_counts=True)) 
y= to_categorical(y)
print(y) #(178,3)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=321, stratify=y)
print(x_train.shape, x_test.shape) 
#(124,13),(54,13)
x_train = x_train.reshape(124,13,1)
x_test = x_test.reshape(54,13,1)
print(y_train.shape) #(124,3)


model= Sequential()
model.add(LSTM(10,activation='relu',input_shape=(13,1)))
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='linear'))
model.add(Dense(3,activation='softmax'))
model.summary()
# ValueError: Shapes (1, 3) and (1, 13) are incompatible
#model.add 여기서 .add 안 넣었더니 생긴 에러


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,batch_size=1,validation_split=0.2,verbose=1)
loss, accuracy = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis =1)

y_test = np.argmax(y_test,axis =1)

acc = accuracy_score(y_test,y_predict)
print(acc)
# 0.3888888888888889
