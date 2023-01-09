import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. datasets
path = './_data/bike/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
sample= pd.read_csv(path+'sampleSubmission.csv', index_col=0)

x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']
print(x.shape, y.shape)  # (506,13)  (506,)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,train_size=0.8,random_state=333)

#2. model
model = Sequential()
model.add(Dense(6, input_dim=8))  
model.add(Dense(12))    
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1))


#3.compile, fit

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode = 'min',patience=20, restore_best_weights=True, verbose = 1)

hist = model.fit(x_train,y_train, epochs=200, batch_size=32
          ,validation_split= 0.2, 
          verbose = 1, callbacks=[earlyStopping]
          )

#4. evaluate,predict
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)
# print("====================================")
# print( hist)  #<tensorflow.python.keras.callbacks.History object at 0x28b74f790>
# print("====================================")
# print(hist.history)
print("====================================")
print(hist.history['loss']) # -> history에서 loss 값만 나옴 
print("====================================")
# print(hist.history['val_loss']) # -> history에서 val_loss 값만 나옴 
# print("====================================")

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],c = 'red', marker = '.', label = 'loss') #x는 epoch로 순서대로 임, 따라서 안 써줘도 됨
plt.plot(hist.history['val_loss'], c='blue', marker = '.',label = 'val_loss' )
plt.grid()  # 격자가 생김(모눈종이형태)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('pandas loss')
plt.legend(loc='upper right') # 그래프에 빨간 선이 뭔지, 파란 선이 뭔지 나옴, 위치는 자동으로 나오고 , 직접 지정도 가능하다. loc = local // upper or down , right of left
plt.show()






