# keras36_dnn1_mnist.py 복붙

import numpy as np
from tensorflow.keras.datasets import mnist

# 1. data
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) // (60000, 28, 28) = (60000, 28, 28, 1)  흑백사진 reshape 해주고 flatten()해줌 
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)

### scaler
x_train = x_train/255.
x_test = x_test/255.


print(x_train.shape) 
print(x_test.shape)  

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#      dtype=int64))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Flatten, Dropout
print('len : ' ,len(x_train)) #len :  60000

#2. model
model = Sequential()
model.add(Dense(128,activation='relu',input_shape=(28,28)))  
model.add(Dropout(0.3))
# model.add(Flatten()) 
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32,activation='linear'))
model.add(Flatten())  # 위치 상관 없음
model.add(Dense(10,activation='softmax'))
model.summary()

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 28, 128)           3712
 dropout (Dropout)           (None, 28, 128)           0
 dense_1 (Dense)             (None, 28, 64)            8256
 dropout_1 (Dropout)         (None, 28, 64)            0
 dense_2 (Dense)             (None, 28, 32)            2080
 flatten (Flatten)           (None, 896)               0                                 => y값이 2차원이니깐
 dense_3 (Dense)             (None, 10)                8970
=================================================================
Total params: 23,018
Trainable params: 23,018
Non-trainable params: 0
_________________________________________________________________
"""




# #3. compile
# model.compile(loss ='sparse_categorical_crossentropy' ,optimizer='adam', metrics='acc')
# hist = model.fit(x_train,y_train, epochs= 30, verbose=1 ,validation_split=0.30, batch_size=32)

# #4. evaluate, predict
# result = model.evaluate(x_test,y_test)
# print('loss : ', result[0]) #[0]처리 안 하면 loss , acc  2개 나옴
# print('acc : ', result[1])  #[1]처리해서 acc 나오게 됨
# print("====================================")
# print(hist.history['val_acc'])  
# print("====================================")



# #결과
# # loss :  0.09270334988832474
# # acc :  0.977400004863739
# # val_acc : 0.9753333330154419
