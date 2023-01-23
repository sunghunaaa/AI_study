import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape)    # (3,10)
print(y.shape)    # (10,)

x = x.T
print(x.shape)    # (10,3)
print(x)

"""
model = Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=1)
loss = model.evaluate(x,y)
print('loss : ', loss)
result = model.predict([[10,1.4,0]]) # **[[10,1.4,0]]
print('result : ', result)
"""
"""
result : 20.06441
loss : 0.14490263
"""

"""
git error 확인 
"""
