import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. data
x = np.array([range(10), range(21, 31), range(201, 211)]) # -> 0부터 10-1 (9)까지
print(x.shape) # (3,10)  range(10)만 있을 때 (1,10)
x = x.T  #10,3

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = y.T #10,2

#2. model

model = Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(5))
model.add(Dense(2))

#compile
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=5)

loss = model.evaluate(x,y)
print('loss : ', loss)
result = model.predict([[9,30,210]])
print('result : ', result)

#result : 10, 1.4

#loss :  0.22765114903450012
#result :  [[9.783808  1.3656275]]

#loss :  0.1596180498600006
#result :  [[10.06123    1.6382837]]

#loss :  0.12498507648706436
#result :  [[9.998747  1.3862599]]

#output_dim=1로 하면 dim=2일 때 중간값으로 감
