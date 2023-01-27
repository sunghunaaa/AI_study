import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


##1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True))
# y값이 label 별로 각각 몇 개인지
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(x.shape, y.shape)  # (178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.1, stratify=y)

print(x_train.shape, x_test.shape) 
#(160, 13) (18, 13)

x_train = x_train.reshape(160,13,1,1)
x_test= x_test.reshape(18,13,1,1)
print(x_train.shape, x_test.shape) #(404,13,1,1) (102,13,1,1)


#2. model

model = Sequential()
model.add(Conv2D(64, kernel_size=(2,1), input_shape=(13,1,1), activation='relu'))
model.add(Flatten()) 
model.add(Dense(3, activation='softmax'))
model.summary()

##3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=30, batch_size=1,
          validation_split=0.125,  verbose=1)

##4. 평가, 예측

print("================== 1. 기본 출력 =================")
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse, ' / mae:', mae)

y_predict = model.predict(x_test)

y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test,axis=1)
from sklearn.metrics import mean_squared_error, accuracy_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

r2 = accuracy_score(y_test, y_predict)
print("R2:", r2)

"""
mse: 0.10902123153209686  / mae: 0.042224153876304626
RMSE: 0.16378897
R2: 0.8776659749432062
"""