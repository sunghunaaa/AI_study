import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error,r2_score

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=13)
print(x_train.shape)



model = Sequential()
model.add(Dense(100,input_dim =10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3.compile, fit

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train,epochs=1000, batch_size=32   # hist = history의 준말 
          ,validation_split= 0.2, 
          verbose = 1
          )

#4. evaluate,predict
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)
print("====================================")
print( hist)  #<tensorflow.python.keras.callbacks.History object at 0x28b74f790>
print("====================================")
print(hist.history)
print("====================================")
print(hist.history['loss']) # -> history에서 loss 값만 나옴 
print("====================================")
print(hist.history['val_loss']) # -> history에서 val_loss 값만 나옴 
print("====================================")

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],c = 'red', marker = '.', label = 'loss') #x는 epoch로 순서대로 임 따라서 안 써줘도 됨
plt.plot(hist.history['val_loss'], c='blue', marker = '.',label = 'val_loss' )
plt.grid()  # 격자가 생김(모눈종이형태)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('diabets loss')
plt.legend(loc='upper right') # 그래프에 빨간 선이 뭔지, 파란 선이 뭔지 나옴, 위치는 자동으로 나오고 , 직접 지정도 가능하다. loc = local // upper or down , right of left
plt.show()

