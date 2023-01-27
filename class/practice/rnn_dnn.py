import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


#1. data
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) # (10,)
x= np.array([[1,2,3],   
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9]])
y= np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7,3) ,(7,)


#2. model
model = Sequential()
model.add(Dense(128, input_shape=(3,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3. compile, fit
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=2000,batch_size=1, verbose=2)


#4. predict
loss = model.evaluate(x,y)
print(loss)
y_pred = np.array([[8,9,10]])
result = model.predict(y_pred)
print(result)
"""
(rnn)
0.051102638244628906
[[10.936562]]
"""
"""
(dnn)
0.005247048102319241
[[11.013697]]
"""
