from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# 1. data
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       #(10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts= True)) 
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
#        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))

#2. model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), input_shape=(32,32,3),
                 activation='relu'))  #31,31,32
model.add(Conv2D(filters=32, kernel_size=(2,2) )) #30,30,32
model.add(Conv2D(filters=32, kernel_size=(2,2) )) #29,29,32 - > 26.912
model.add(Flatten())
model.add(Dense(320,activation='relu'))
model.add(Dense(100,activation='softmax'))

#3. compile
mcp = ModelCheckpoint(
    monitor="val_loss",
    mode="auto",
    save_best_only=True,
    filepath= './_data/cifar100/cifar_modelcheckpoint.hdf5',
    verbose =1
)
es = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=1)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(x_train,y_train,epochs=20, verbose=1,validation_split=0.2, batch_size= 32,
                 callbacks=[es,mcp])

#4.predict
result = model.evaluate(x_test,y_test)

