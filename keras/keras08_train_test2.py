import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array(range(10))

#실습 : 넘파이 리스트 슬라이싱!! 7:3으로 잘라내기!! 
#배열[a:b:c]를 이용하여 배열의 일부를 잘라 표시할 수 있습니다.
#a는 시작값, b는 도착값, c는 간격을 의미합니다.

#동일하게 배열[a:b:e, c:d:f]를 이용하여 e와 f를 간격으로 사용할 수 있습니다.
#a ~ b는 표시할 행의 위치를 의미하며, c ~ d는 표시할 열의 위치를 의미합니다.
x_train = x[:7]  # [:-3] 
x_test = x[7:]  # [-3:]
y_train = y[:7]
y_test = y[7:]

# [ : : 1] 범위를 전체로 두고 train에 간격을 1로 둔다. weight 값을 위해
# 

print(x_train)
print(x_test)
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
result = model.predict([11])
print('result : ', result)
"""