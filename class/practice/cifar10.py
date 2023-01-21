from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# 1. data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       #(10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts= True)) 
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
# array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#      dtype=int64))

#2. model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), input_shape=(32,32,3),
                 activation='relu'))  #31,31,32
model.add(Conv2D(filters=32, kernel_size=(2,2) )) #30,30,32
model.add(Conv2D(filters=32, kernel_size=(2,2) )) #29,29,32 - > 26.912
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3. compile
mcp = ModelCheckpoint(
    monitor="val_loss",
    mode="auto",
    save_best_only=True,
    filepath= './_data/cifar10/cifar_modelcheckpoint.hdf5',
    verbose =1
)
es = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=1)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(x_train,y_train,epochs=20, verbose=1,validation_split=0.2, batch_size= 32,
                 callbacks=[es,mcp])

#4.predict
result = model.evaluate(x_test,y_test)

print(mcp)
