from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten
import numpy as np

##1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7,
    random_state=9)


print(x_train.shape, x_test.shape) 
#(14447, 8) (6193, 8)

x_train = x_train.reshape(14447,4,2,1)
x_test= x_test.reshape(6193,4,2,1)
print(x_train.shape, x_test.shape) 

#2. model

model = Sequential()
model.add(Conv2D(64, kernel_size=(2,2), input_shape=(4,2,1), activation='relu'))
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
