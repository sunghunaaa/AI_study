from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten
import numpy as np


##1. 데이터
datasets = load_diabetes()
print(datasets)
x = datasets.data
y = datasets.target
print(x.shape) #(442, 10)
print(y.shape) #(442,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7,
    random_state=9)

print(x_train.shape, x_test.shape) #(309, 10) (133, 10)
print(x_train)

x_train = x_train.reshape(309,5,2,1)
x_test= x_test.reshape(133,5,2,1)
print(x_train.shape, x_test.shape) 

#2. model

model = Sequential()
model.add(Conv2D(64, kernel_size=(2,2), input_shape=(5,2,1), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.summary()

##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=10, batch_size=1,
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

# mse: 4523.31494140625  / mae: 57.36675262451172
# RMSE: 67.25559589510631
# R2: 0.18148898497935106

# mse: 4524.06640625  / mae: 57.42216110229492
# RMSE: 67.26117763578644
# R2: 0.18135311804502408
