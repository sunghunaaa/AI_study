#실습                         
# 1.R2를 음수가 아닌 0.5 이하로 줄이기
# 2.데이터는 건들지 말 것
# 3. 레이어는 인풋 아웃풋 포함 7개 이상
# 4. batch_size = 1
# 5. 히든 레이어의 노드는 각각 10개 이상 100개 이하
# 6. train 70%
# 7. epoch 100번 이상
# 8. loss 지표는 mse 또는 mae

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score  # 한 개의 sklearn.metrics에 여러개 import 하는 경우


x = np.array(range(1,21))
y = np.array(range(1,21))

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.7,shuffle=True)
model = Sequential()   # input, output 제외한 노드를 포함한 레이어를 히든 레이어라고 함
model.add(Dense(100, input_dim=1))                                                    
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1)) 

model.compile(loss='mae', optimizer='adam', metrics=['mae'])  
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print("====================")
print(y_test)
print(y_predict)
print("====================")


# def - 정의할 때 사용
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   #->mean_squared_error(y_test, y_predict) = 'mae', sqrt=> 루트

print("RMSE : ", RMSE(y_test,y_predict))
  
r2= r2_score(y_test,y_predict)
print("R2 : ", r2)


# RMSE :  3.607839252146836
# R2 :  0.31190286858159777














