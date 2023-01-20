
import numpy as np
from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

#1. data
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)  #(1797, 64) (1797,)
print(np.unique(y,return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
# #array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

y= to_categorical(y)
print(y.shape) #(1797,10)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=321, stratify=y)


model= Sequential()
model.add(Dense(10,activation='relu',input_dim=64))
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='linear'))
model.add(Dense(10,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=46,batch_size=1,validation_split=0.2,verbose=1)
loss, accuracy = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis =1)

y_test = np.argmax(y_test,axis =1)

acc = accuracy_score(y_test,y_predict)
print(acc)










# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[3])
# plt.show()
