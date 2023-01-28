# 큰데이터 수작업 못 하는 데이터들 5개씩 자르는 거 // [1,2,3,4,5],[2,3,4,5,6] ... ,[6,7,8,9,10]  이렇게 split해줌

import numpy as np

a = np.array(range(1, 11))
timesteps = 5 
# timesteps 몇 개씩 자를 건지
# 5라면 [1,2,3,4,5],[2,3,4,5,6], ... ,[6,7,8,9,10]
# 3이라면 [1,2,3],[2,3,4],[3,4,5], ... ,[8,9,10]
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1 ):  # 만약 range(3) = (0,1,2)
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset) # 뒤에 추가로 쳐 박는다.
    return np.array(aaa)
bbb = split_x(a, timesteps)
print(bbb) 
"""
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
"""
print(bbb.shape) #(6, 5)

x = bbb[:, :-1]  # [행 , 열]
y = bbb[:, -1]
print(x)
print(y)
print(x.shape,y.shape) # (6, 4) (6,)
#실습
#LSTM 모델 구성
x_predict= np.array([7,8,9,10])
x = x.reshape(6,4,1)


#2.model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()
model.add(LSTM(128, input_shape=(4,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3.compile, fit
model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)

#4. evaluate
loss = model.evaluate(x,y)
x_predict = x_predict.reshape(1,4,1)
y_predict = model.predict(x_predict)
print(y_predict)
#[[10.940507]]
