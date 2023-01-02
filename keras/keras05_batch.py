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
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=10, batch_size=2) 
# 배치사이즈 1 일경우 6
# 배치사이즈 2 일경우 3
# 배치사이즈 3 일경우 2
# 배치사이즈 4 일경우 2
# 배치사이즈 5 일경우 2
# 배치사이즈 6 일경우 1
# 배치사이즈 7 일경우 1
# 배치사이즈 8 일경우 1
# 배치사이즈의 디폴트 값은 32 


result = model.predict([6])
print('결과 : ', result)

"""
(블럭 주석)
"""

"""
배치사이즈 1 일경우 6
배치사이즈 2 일경우 3
배치사이즈 3 일경우 2
배치사이즈 4 일경우 2
배치사이즈 5 일경우 2
배치사이즈 6 일경우 1
배치사이즈 7 일경우 1
배치사이즈 8 일경우 1
배치사이즈의 디폴트 값은 32 
"""

