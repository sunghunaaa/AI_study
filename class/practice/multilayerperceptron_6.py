import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([range(10)])                          # (10,) (10,1) dim이 동일하게 작용함
print(x.shape) #(1,10)
x = x.T #(10,1)
print(x.shape)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])
print(y.shape) #3,10
y = y.T #10,3
print(y.shape)

model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(100))
model.add(Dense(2000))
model.add(Dense(100))
model.add(Dense(3))

model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=5)

loss = model.evaluate(x,y)
print('loss : ', loss)
result = model.predict([9])  #-> 열이 1이기만하면 되니깐 [[9]] [9] 둘 다 가능!
print('result : ', result)

#loss :  0.05092155933380127
#result :  [[9.989053   1.6354787  0.01473856]]

#결과값 dim 3인 상황에서 output_dim=2로 바꿨을 때 error/ output_dim=1로 바꿨을 때 result 값 1.6795, loss 값 2.4678
