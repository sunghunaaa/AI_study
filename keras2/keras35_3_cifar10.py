from tensorflow.keras.datasets import cifar10
import numpy as np
from sklearn.model_selection import train_test_split


# 1. data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#       dtype=int64))

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
model.add(Conv2D(filters=64,kernel_size= (2,2),padding='same', activation='relu')) 
model.add(Conv2D(filters=64,kernel_size= (2,2),padding='same',  activation='relu')) 
model.add(Flatten()) 
model.add(Dense(32,activation='relu')) 
model.add(Dense(10,activation='softmax'))

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

# loss : 2.302626132965088
# acc : 0.10000000149011612
# val_acc : 0.09839999675750732