import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


##1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape)
y = to_categorical(y)   # 원핫 인코딩 (1797, 64) (1797,)
print(x.shape, y.shape) # (1797, 64) (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.2, stratify=y)

print(x_train.shape, x_test.shape) 
#(1437, 64) (360, 64)

x_train = x_train.reshape(1437,8,8,1)
x_test= x_test.reshape(360,8,8,1)
print(x_train.shape, x_test.shape) #(404,13,1,1) (102,13,1,1)


#2. model

model = Sequential()
model.add(Conv2D(64, kernel_size=(2,2), input_shape=(8,8,1), activation='sigmoid'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

##3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1, batch_size=1,
          validation_split=0.125,  verbose=1)

##4. 평가, 예측
print("x_test",x_test)
print("x_train",x_train)
print("y_test",y_test)
print("y_train",y_train)


print("================================")
mse, mae = model.evaluate(x_test, y_test)
print('loss:', mse, ' / acc:', mae)


y_predict = model.predict(x_test)
y_predict=np.argmax(y_predict,axis=1)

y_test=np.argmax(y_test,axis=1)

from sklearn.metrics import accuracy_score
acc= accuracy_score(y_test,y_predict)
print(acc)
