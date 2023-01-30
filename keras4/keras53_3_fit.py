import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  #전처리
#1.data
train_datagen = ImageDataGenerator(
    rescale=1./255. ,
    horizontal_flip=True, #수평 반절한다는 거
    vertical_flip= True,#수직 반절한다는 거
    width_shift_range=0.1, #이동
    height_shift_range=0.1, #이동
    rotation_range=5, #회전하겠다는 거
    zoom_range=1.2, #확대
    shear_range=0.7,
    fill_mode='nearest',  #채우기
)

test_datagen= ImageDataGenerator(
    rescale=1./255.
)
# 일반 fit을 사용하면 xy가 합쳐진 데이터가 아닌 x,y가 분리된 데이터를 사용해야한다. 
#xy_train[0][0],xy_train[0][1]
 
xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',    
    target_size=(100,100),
    batch_size= 1000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',    
    target_size=(100,100),
    batch_size= 1000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64,(2,2), input_shape=(100,100,1))) #99,99,64
model.add(Conv2D(64,(3,3),activation='relu')) #97,97,64
model.add(Conv2D(32,(3,3)   ,activation='relu')) #95,95,32
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1,activation='sigmoid'))  # softmax 쓰려면 output_dim = 2여야 해

#3. compile, fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(xy_train[0][0],xy_train[0][1],epochs=13,validation_data=(xy_test[0][0],xy_test[0][1]),
                 batch_size=16,
                 )  

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
[0.7175647020339966, 0.6931644678115845, 0.6931567788124084, 0.6931860446929932, 0.6931549310684204, 0.6931876540184021, 0.6931799650192261, 0.6932042837142944, 0.6931495070457458, 0.6931589841842651, 0.6931495070457458, 0.6931607127189636, 0.6931571364402771]
loss :  0.6931571364402771
val_loss :  0.6931480169296265
accuracy :  0.5
val_acc :  0.5
"""