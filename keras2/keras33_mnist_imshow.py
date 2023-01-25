import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


# 1. data
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) // (60000, 28, 28) = (60000, 28, 28, 1)  흑백사진 reshape 해주고 flatten()해줌 
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

print(x_train.shape)  #(60000, 28, 28, 1)  1늘어나도 data의 성질이 바뀌지 않은 cnn에 넣기 위해 4차원으로 바꿨을 뿐
print(x_test.shape)  #(10000, 28, 28, 1) 

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
print('len : ' ,len(x_train))

import matplotlib.pyplot as plt
plt.imshow(x_train[1], 'gray')
plt.show()


#conv2d 4차원 dense는 2차원

#2. model
model = Sequential()
model.add(Conv2D(filters=128, kernel_size= (2,2), input_shape=(28,28,1),
                 activation='relu'))    #(27,27,128)
model.add(Conv2D(filters=64,kernel_size= (2,2)))  #(26,26,64)
model.add(Conv2D(filters=64,kernel_size= (2,2)))  #(25,25,64)  -> 40000
model.add(Flatten())  # -> 40000
model.add(Dense(32,activation='relu'))  #(60000,40000)//input_shape = (40000,)  60000은 batch_size 40000은 input_dim
model.add(Dense(10,activation='softmax'))

#3. compile
from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor="loss",
    mode="auto",
    save_best_only=True,
    filepath= ("test.hdf5"),
    verbose=1 
)

model.compile(loss ='sparse_categorical_crossentropy' ,optimizer='adam', metrics='acc')
hist = model.fit(x_train,y_train, epochs= 2, verbose=1 ,validation_split=0.30, batch_size=32, callbacks=[mcp])

#4. evaluate, predict
result = model.evaluate(x_test,y_test)
print('loss : ', result[0]) #[0]처리 안 하면 loss , acc  2개 나옴
print('acc : ', result[1])  #[1]처리해서 acc 나오게 됨



#earlystopping , model check point 적용 val 적용
