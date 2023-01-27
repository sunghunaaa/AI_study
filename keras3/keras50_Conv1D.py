#keras49_Bidirectional2 복붙
from tensorflow.keras.layers import Dense,LSTM ,Bidirectional, GRU
from tensorflow.keras.models import Sequential
#############################################################################
from tensorflow.keras.layers import Conv1D
#############################################################################


import numpy as np
a = np.array(range(1,101))
x_predict = np.array(range(96,106)) # 예상 y= 100, 106

timesteps1 = 5  
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1 ):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset) 
    return np.array(aaa)
bbb = split_x(a, timesteps1)
ccc = split_x(x_predict, 4)
print(bbb.shape) #(96.5)
print(ccc)

x = bbb[:, :-1]  # [행 , 열]
y = bbb[:, -1]
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,shuffle=True ,random_state=321)   
x_train = x_train.reshape(72,4,1)
x_test = x_test.reshape(24,4,1)

x_predict = ccc.reshape(7,4,1)

#2.model

model = Sequential()
model.add(Conv1D(100,4,input_shape=(4,1))) 
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv1d (Conv1D)             (None, 3, 100)            300       => 연산량 굉장히 낮음
"""

#model.add(LSTM(100, input_shape=(4,1)))
# """
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 100)               40800
# """                                                      



model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.summary()
#3.compile, fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=200,batch_size=1,validation_split=0.1)

#4. evaluate
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_predict)
print(y_predict)
"""
[[[100.035194]]

 [[101.035515]]

 [[102.035835]]

 [[103.03615 ]]

 [[104.036476]]

 [[105.0368  ]]

 [[106.03712 ]]]
"""