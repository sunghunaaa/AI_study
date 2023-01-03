import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. data
x = np.array([range(10), range(21, 31), range(201, 211)])
x = x.T  
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = y.T 

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.3,
    random_state=123
)
print('x_tr :',x_train)
print('x_t :',x_test)
print('y_tr :',y_train)
print('y_t :',y_test)


#실습    train_test_split를 이용하여
#7:3으로 잘라서 모델 구현 / 소스 완성
#model
model =Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(2))

#compile
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train,epochs=200,batch=1)

#result, loss
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)
result = model.predict([[0,21,201]])
print('result : ', result)
