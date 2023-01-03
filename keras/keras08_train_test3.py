import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
#1. data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array(range(10))

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    #train_size=0.7, 70%
    test_size=0.3,
    #shuffle= False,
    random_state=123  # random_state에 계속 같은 난수 ex)123입력하면 몇 번을 돌리도 동일한 값 나 옴)
)

# x_train = x[:7]  # [:-3] 
# x_test = x[7:]  # [-3:]
# y_train = y[:7]
# y_test = y[7:]

# [검색] train과 test를 섞어서 7:3으로 만들어 (사이킷 런)

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