import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.7, random_state=123)
                                                 
model = Sequential()
model.add(Dense(10, input_dim=1))                                                    
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) 

model.compile(loss='mse', optimizer='adam', 
              metrics=['mae','mse','accuracy','acc'])  # -> 훈련 중 가중치에 영향을 줌 /metrics는 훈련에 영향을 주지 않고 참고용 지표로 사용 됨.([]안에는 2개이상 쓸 수 있다.)
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

# mae : 3.1518757343292236
# mse : 14.67847728729248