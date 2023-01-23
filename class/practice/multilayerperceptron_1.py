# mlp = multi layer perceptron 
# 행무시, 열우선 
# 열 = 특성, 컬럼, 피쳐, input_dim = 열 (행,열)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape) # (2, 10)  
print(y.shape) # (10,)

x = x.T # T 전치 = 행과 열을 바꿔준다.
print(x.shape) # (10, 2) 

#2. model
model = Sequential()
model.add(Dense(5,input_dim=2))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. compile, fit
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=1)

#4. evaluate, result - evaluate 배치 사이즈 default(32) 값임 
loss = model.evaluate(x,y) # -> 훈련시킨 범위 내에서 loss값임으로 결과가 좋아보이지만, 범위를 벗어나면 loss값 안 좋아짐
print('loss : ', loss)     # -> 100% 중 70% 데이터로 훈련시키고 나머지 30%로 평가를 시킨다. 예) x값으로 1~7까지 훈련 나머지 평가로 사용한다는 뜻임.

result = model.predict([[10,1.4]])
print('result : ', result)

"""
loss : 0.0737300
result : 19.97016
"""

