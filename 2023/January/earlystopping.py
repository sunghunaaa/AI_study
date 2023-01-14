from sklearn.datasets import load_boston
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=321)
#2. model
model = Sequential()
model.add(Dense(10,activation='relu',input_shape=(13,)))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='relu'))
#3. compile,fit
model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(monitor='val_loss', mode = 'min', patience=20, restore_best_weights= True, verbose=1)
"""
val_loss를 모니터하는데 의문을 갖으면 안 됨. 당연한 것/ history의 val_loss의 최소값을 이용함.(현재 mode = min이라서 최소값)
mode - 'min', 'max', 'auto'(accracy 사용 시에는 정확도 높을 수록 좋음 따라서 max로 설정함.)
patience - 횟수만큼 val_loss값이 역전되면 훈련을 멈춘다.
restore_best_weights - 횟수만큼 진행했을 때 가장 val_loss 값이 작은 weight 값의 위치를 저장해둔다./ True로 설정해둬야 종료 시점의 weight값이 아닌 최적의 weight값을 사용 가능하다.
verbose - fit에서 사용되던 때와 유사한 기능
"""
hist = model.fit(x,y,epochs=__,batch_size=32,
                 validation_split=0.2,
                 verbose=1,
                 callbacks=[earlyStopping])
# callbacks - earlyStopping한 위치를 다시 불러옴
#4. evaluate,predict
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)


import matplotlib.pyplot as plt
plt.figure(figsize = (9,6))
plt.plot(hist.history['val_loss'], c='blue', marker='.',label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('boston loss')
plt.legend(loc = 'upper right')
plt.show()
