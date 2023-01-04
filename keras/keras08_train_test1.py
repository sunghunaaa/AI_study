import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
# x = np.array([1,2,3,4,5,6,7,8,9,10]) # (10,)
# y = np.array(range(10))  # 0~9         (10,)
x_train = np.array([1,2,3,4,5,6,7])         #(7,)
x_test = np.array([8,9,10])                  #(3,)
y_train = np.array(range(7))
y_test = np.array(range(7,10)) 


print(y_train)
print(y_test)


"""
#2. model
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(20))
model.add(Dense(100))
model.add(Dense(40))
model.add(Dense(1))

#compile
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train,epochs=1000, batch_size=1)

#evaluate

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)
result = model.predict( )
print('result : ', result)

#loss :  0.17393667995929718
#result :  [[10.213268]]

#loss :  0.03468036651611328
#result :  [[10.041458]]

"""