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
 
xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',    
    target_size=(100,100),
    batch_size= 10,
    class_mode='binary',
    color_mode='grayscale', #class_mode : 분류 방식에 대해서 지정합니다. categorical : 2D one-hot 부호화된 라벨이 반환됩니다.,binary : 1D 이진 라벨이 반환됩니다.,sparse : 1D 정수 라벨이 반환됩니다.,None : 라벨이 반환되지 않습니다.
    shuffle=True,
)

print(xy_train.class_indices) # {'ad': 0, 'normal': 1}
xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',    
    target_size=(100,100),
    batch_size= 10,
    class_mode='binary',
    color_mode='grayscale', #class_mode : 분류 방식에 대해서 지정합니다. categorical : 2D one-hot 부호화된 라벨이 반환됩니다.,binary : 1D 이진 라벨이 반환됩니다.,sparse : 1D 정수 라벨이 반환됩니다.,None : 라벨이 반환되지 않습니다.
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
hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=13
                    ,validation_data=xy_test,
                    validation_steps=16,)  #batch 없어도 됨, 위에서 했음

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
[0.763585090637207, 0.6893177032470703, 0.7206422686576843, 0.6907913088798523, 0.6931888461112976, 0.6929699182510376, 0.6927604079246521, 0.6941468119621277, 0.6929694414138794, 0.6953426599502563, 0.6840404868125916, 0.7182789444923401, 0.6962069272994995]
loss :  0.6962069272994995
val_loss :  0.6899295449256897
accuracy :  0.5
val_acc :  0.5
"""
# 그림 그려보기 

