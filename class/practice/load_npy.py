import numpy as np
"""
np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])
#np.save('./_data/brain/xy_train.npy', arr=xy_train[0]) =>tuble 형태라 안 됨// x,y따로 load해서 append로 합쳐줘야 함 keras47_split1.py참고
np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])

"""

x_train = np.load('./_data/brain/brain_x_train.npy')
y_train = np.load('./_data/brain/brain_y_train.npy')
x_test = np.load('./_data/brain/brain_x_test.npy')
y_test = np.load('./_data/brain/brain_y_test.npy')
 
print(x_train.shape, x_test.shape) #(160, 200, 200, 1) (120, 200, 200, 1)
print(y_train.shape, y_test.shape) #(160,) (120,)

print(x_train[100])


#2. model 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64,(2,2), input_shape=(200,200,1))) #99,99,64
model.add(Conv2D(64,(3,3),activation='relu')) #97,97,64
model.add(Conv2D(32,(3,3)   ,activation='relu')) #95,95,32
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1,activation='sigmoid'))  # softmax 쓰려면 output_dim = 2여야 해

#3. compile, fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train,epochs=13,validation_data=(x_test,y_test),
                 batch_size=16,
                 )  #fit_generator가 아니라서 batch_size 필수

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print(loss)  # epchs 갯수만큼 나옴, 훈련때에 나왔던 loss 값 순서대로 다 나옴
print('loss : ', loss[-1]) #'훈련의' 가장 마지막 loss값 나옴
print('val_loss : ', val_loss[-1]) #'훈련의' 가장 마지막 val_loss값 나옴
print('accuracy : ', accuracy[-1]) #'훈련의' 가장 마지막 accuracy값 나옴
print('val_acc : ', val_acc[-1]) #'훈련의' 가장 마지막 val_acc값 나옴

"""
loss :  0.00849272683262825
val_loss :  0.0258196834474802
accuracy :  1.0
val_acc :  0.9833333492279053
"""
