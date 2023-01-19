import tensorflow as tf 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. model
model = Sequential()  # => model 정의함.
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. compile
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=200, batch_size=7) 
# -> y=wx+b 의 w값과 b값 결정 됨.


#4. evaluate, predict
loss = model.evaluate(x,y)  #-> 가장 마지막 loss값이 평가값이 됨
print('loss : ', loss)
result = model.predict([6])
print('결과 : ', result)


# 결과 값 : 5.880807 , loss : 0.370444536209 
# loss 값이 기준이다. 
# 결과 값 : 6 이 나와야 한다는 것을 알고있고, 6과 결과 값 차이보다 loss 값이 더 큰 상관이다.
# 그래도 기준은 loss가 되어야 함.
