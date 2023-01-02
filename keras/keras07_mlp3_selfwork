#import
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#data
x = np.array([[1,2,3,4,5,6,7,8,9,10],[9,8,7,6,5,4,3,2,1,0],[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape)    # 3,10
print(y.shape)    # 10,0

x = x.T
print(x.shape)    # 10,3

#model
model = Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


#compile,fit
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=200,batch_size=1)

#evaluate,result
loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([[10,0,2.0]])
print('result : ', result)      

"""
loss : 0.0184999
result : 19.992905
"""
