import numpy as np
x_train = np.load('C:/_data/dogcat_num/x_train.npy')
y_train = np.load('C:/_data/dogcat_num/y_train.npy')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,MaxPool2D
from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test = train_test_split(x_train,y_train,test_size=0.2,random_state=321,shuffle=True)
model = Sequential()
model.add(Conv2D(32,(5,5), input_shape=(150,150,3))) #99,99,64
model.add(MaxPool2D())
model.add(Conv2D(32,(5,5),activation='relu')) #95,95,32
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = 'C:/_data/mcp/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor="val_acc",
    mode="auto",
    save_best_only=True,
    filepath= filepath+'dogcat_'+date+'_'+filename,
    verbose =1
)

hist = model.fit(x1_train, y1_train,epochs=2000, validation_split=0.1,
                 batch_size=32, callbacks=[mcp])


accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print(loss)  # epchs 갯수만큼 나옴, 훈련때에 나왔던 loss 값 순서대로 다 나옴
print('loss : ', loss[-1]) #'훈련의' 가장 마지막 loss값 나옴
print('val_loss : ', val_loss[-1]) #'훈련의' 가장 마지막 val_loss값 나옴
print('accuracy : ', accuracy[-1]) #'훈련의' 가장 마지막 accuracy값 나옴
print('val_acc : ', val_acc[-1]) #'훈련의' 가장 마지막 val_acc값 나옴
