#keras36_dnn3_cifar10.py

from tensorflow.keras.datasets import cifar10
import numpy as np
from sklearn.model_selection import train_test_split


# 1. data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

#### flatten
x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3) 

### scaler
x_train = x_train/255.
x_test = x_test/255.

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#       dtype=int64))

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, Input


#2. model(함수형)
input1 = Input(shape=(3072,))
dense1 = Dense(128, activation='relu')(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(20, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(10, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)

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

# loss :  3.3607540130615234
# acc :  0.6226000189781189
# val_acc :  0.6251333355903625




#dnn 결과값
# loss :  1.939517617225647
# acc :  0.2599000036716461
# val_acc : 0.2524000108242035

#
# loss :  1.5083162784576416
# acc :  0.46050000190734863
# val_acc : 0.45473334193229675