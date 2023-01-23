import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[1,2,3],[1.1,1.2,1.3]])
y = np.array([3,6,9])

print(x.shape)  # 2,3
print(y.shape)  # 3,0

x = x.T
print(x.shape)  # 3,2

model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x,y,epochs=200, batch_size=1)

loss = model.evaluate(x,y)
result = model.predict([[2,1.2]])

print('loss : ', loss)
print('result : ', result)

# loss 0.01655
# result 6.0165
