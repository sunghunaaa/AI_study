#error

from sklearn.datasets import load_iris
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##1. 데이터
datasets = load_iris()

print(datasets.feature_names)   #  pandas에서는 .columns

x = datasets.data
y = datasets['target']
print(x.shape)  # (150, 4)
print(y.shape)  # (150,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.2,
    stratify=y 
)
print(x_train.shape, x_test.shape) 
#(120, 4) (30, 4)

x_train = x_train.reshape(120,2,2,1)
x_test= x_test.reshape(30,2,2,1)
print(x_train.shape, x_test.shape) 

#2. model

model = Sequential()
model.add(Conv2D(64, kernel_size=(2,2), input_shape=(2,2,1), activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()

##3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=1,
          validation_split=0.125,  verbose=1)

##4. 평가, 예측

print("================== 1. 기본 출력 =================")
loss, accurac= model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score
