import tensorflow as tf
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 모델 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

models = Sequential()
models.add(Dense(1, input_dim=1))

#컴파일과 훈련

models.compile(loss='mae', optimizer='adam')
models.fit(x,y, epochs = 2000)


result = models.predict([13])
print(result)
