import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255. ,
)
#데이터 저장할 때는 증강된 데이터가 아닌 원본을 들고 있는 것이 좋음
#이미지를 숫자로 변환하는 시간 오래걸림 따라서 수치로 변환된 데이터 끌어다 는는 게 더 빠름
#데이터 크키는 수치가 이미지보다 더 큼

test_datagen= ImageDataGenerator(
    rescale=1./255.
)
 
xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',    
    target_size=(200,200),
    batch_size= 100000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',    
    target_size=(200,200),
    batch_size= 100000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])
#np.save('./_data/brain/xy_train.npy', arr=xy_train[0])

np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])
