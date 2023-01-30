
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split


#1. data
datasets = load_iris()
x = datasets.data
y = datasets['target']
#print(x.shape,y.shape) # (150,4) (150,)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,
                        shuffle=True,
                      random_state= True,
                      test_size= 0.2,
                      stratify=y)  
print(x_train.shape) #(120,4)
print(x_test.shape) #(30,4)
print(y_train.shape) #(120,3)
print(y_test.shape) #(30,3)

x_train = x_train.reshape(120,4,1)
x_test = x_test.reshape(30,4,1)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(4,1)))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax')) 

#compile
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=4, batch_size=1, validation_split=0.2, verbose =1)

loss, accuracy = model.evaluate(x_test, y_test)

from sklearn.metrics import accuracy_score
import numpy as np
print(y_test.shape)  

y_predict = model.predict(x_test) 
print(y_predict.shape) 
y_predict = np.argmax(y_predict, axis =1 )
print(y_predict)
print(y_predict.shape) 



print(y_test) 
print(y_test.shape)
y_test = np.argmax(y_test, axis=1)
print(y_test) 
print(y_test.shape) 


acc = accuracy_score(y_test,y_predict)
print(acc) 

# 0.8666666666666667


