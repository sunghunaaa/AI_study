

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score,accuracy_score
import numpy as np

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split (
    x, y, shuffle=True,
    random_state=333, test_size=0.2
)

print(x.shape, y.shape)  # (569, 30) (569,)

print(x_train.shape, x_test.shape) 
#(455, 30) (114, 30)

x_train = x_train.reshape(455,6,5,1)
x_test= x_test.reshape(114,6,5,1)
print(x_train.shape, x_test.shape) #(404,13,1,1) (102,13,1,1)


#2. model

model = Sequential()
model.add(Conv2D(64, kernel_size=(2,2), input_shape=(6,5,1), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

##3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=1,
          validation_split=0.125,  verbose=1)

##4. 평가, 예측

print("================== 1. 기본 출력 =================")
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

"""
0.22409188747406006
0.9035087823867798
"""
