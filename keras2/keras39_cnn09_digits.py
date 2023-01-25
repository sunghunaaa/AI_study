import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


##1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape)
y = to_categorical(y)   # 원핫 인코딩
print(x.shape, y.shape) # (1797, 64) (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.2, stratify=y)

print(x_train.shape, x_test.shape) 
#(455,13)(51,13)

x_train = x_train.reshape(455,13,1,1)
x_test= x_test.reshape(51,13,1,1)
print(x_train.shape, x_test.shape) #(404,13,1,1) (102,13,1,1)


#2. model

model = Sequential()
model.add(Conv2D(64, kernel_size=(2,1), input_shape=(13,1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.summary()

##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=30, batch_size=1,
          validation_split=0.125,  verbose=1)

##4. 평가, 예측

print("================== 1. 기본 출력 =================")
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse, ' / mae:', mae)

y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2:", r2)