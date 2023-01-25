from tensorflow.keras.datasets import cifar100
import numpy as np
from sklearn.model_selection import train_test_split


# 1. data
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))
"""
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
       68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
       85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
print('len : ' ,len(x_train)) #len :  50000

### scaler
x_train = x_train/255.
x_test = x_test/255.
# """
# import matplotlib.pyplot as plt
# plt.imshow(x_train[1], 'gray')
# plt.show()
# =>파일 그림 볼 때 사용
# """

# #conv2d 4차원 dense는 2차원

#2. model
model = Sequential()
model.add(Conv2D(filters=128, kernel_size= (2,2), input_shape=(32,32,3),
                 padding='same',
                 activation='relu')) 
model.add(MaxPool2D())
model.add(Conv2D(filters=64,kernel_size= (2,2),padding='same')) 
model.add(Conv2D(filters=32,kernel_size= (2,2),padding='same')) 
model.add(Flatten()) 
model.add(Dense(16,activation='relu')) 
model.add(Dense(100,activation='softmax'))

#3. compile
model.compile(loss ='sparse_categorical_crossentropy' ,optimizer='adam', metrics='acc')
hist = model.fit(x_train,y_train, epochs= 30, verbose=1 ,validation_split=0.30, batch_size=32)

#4. evaluate, predict
result = model.evaluate(x_test,y_test)
print('loss : ', result[0]) #[0]처리 안 하면 loss , acc  2개 나옴
print('acc : ', result[1])  #[1]처리해서 acc 나오게 됨
print("====================================")
print(hist.history['val_acc'])  
print("====================================")

#결과값
# loss :  nan
# acc :  0.009999999776482582
# val_acc : 0.010400000028312206