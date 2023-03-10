import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

# 1. data
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) // (60000, 28, 28) = (60000, 28, 28, 1)  흑백사진 reshape 해주고 flatten()해줌 
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

#### flatten
x_train = x_train.reshape(60000,28*28) #(60000, 784)
x_test = x_test.reshape(10000,28*28) #(10000, 784)

### scaler
x_train = x_train/255.
x_test = x_test/255.

print(x_train.shape)  #(60000, 28, 28, 1)  1늘어나도 data의 성질이 바뀌지 않은 cnn에 넣기 위해 4차원으로 바꿨을 뿐
print(x_test.shape)  #(10000, 28, 28, 1) 

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten ,Dropout
print('len : ' ,len(x_train)) #60000

#2. model
model = Sequential()
model.add(Dense(128,activation='relu',input_shape=(784,)))  
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32,activation='linear'))
model.add(Dense(10,activation='softmax'))

#3. compile
model.compile(loss ='sparse_categorical_crossentropy' ,optimizer='adam', metrics='acc')
hist = model.fit(x_train,y_train, epochs= 30, verbose=1 ,validation_split=0.30, batch_size=32)

#4. evaluate, predict
result = model.evaluate(x_test,y_test)
print('loss : ', result[0]) #[0]처리 안 하면 loss , acc  2개 나옴
print('acc : ', result[1])  #[1]처리해서 acc 나오게 됨
print("====================================")
print(hist.history['val_loss'])  
print("====================================")

#결과값
# loss :  0.3591788113117218
# acc :  0.8774999976158142
# val_acc: 0.8829
