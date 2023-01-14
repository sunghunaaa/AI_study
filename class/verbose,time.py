from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. datasets
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (506,13)  (506,)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,train_size=0.8,random_state=333)

#2. model
model = Sequential()
model.add(Dense(5, input_dim=13))  #input_dim은 행과 열 ( , )에서만 사용 가능   
# ex) (100, 10 , 5)  -> (10,5) 가 100개 따라서 행의 역할을 하는 것은 100임. (실제 행으로 불리진 않음) 따라서 input_shape = (10,5)
model.add(Dense(5, input_shape=(13,)))    # input_shape(___,___)  // input_dim =13 과 input_shape=(13,) 동일함
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


#3.compile, fit

model.compile(loss='mse', optimizer='adam')
import time
start = time.time()
model.fit(x_train,y_train,epochs=50, batch_size=1
          ,validation_split= 0.2, 
          verbose = 1
          )
end = time.time()

print("걸린시간 : ", end -start)
#4. evaluate,predict
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)
