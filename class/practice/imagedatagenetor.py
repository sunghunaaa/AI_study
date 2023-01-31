import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  #전처리

# 단어 하나하나 검색 후 뜻으로 유추해보기
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
 
#directory : 폴더 #폴더에 있는 데이터를 가져올 건데 위에 정의해둔 방식으로 가져올 것이다.
#x= 160,150,150,1 이미지 y=(160,0) ,ad = 0 , normal=1
xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',    
    target_size=(200,200),
    batch_size= 10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    #Found 160 images belonging to 2 classes.
)

"""
xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',    
    target_size=(200,200),
    batch_size= 999999,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    #Found 160 images belonging to 2 classes.
)
내가 데이터의 갯수를 모를 때
batch_size =99999로 하고 print(xy_train[0][0].shape) 찍으면  (160, 200, 200, 1) 
데이터의 갯수인 160을 알 수 있다.
"""

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',    
    target_size=(200,200),
    batch_size= 10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    #Found 120 images belonging to 2 classes.
)


print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x00000202D2B8D190>
print(xy_train.shape)
"""
***잘 활용해야할 것들***
print(xy_train[0])
#  batch_size= 10일 때, y 10개나옴
#  batch_size= 3일 때, y 3개 나옴

print(xy_train[0][0]) #x만 나옴
print(xy_train[0][0].shape) # (10, 200, 200, 1), 여기서 10의 batch_size
print(xy_train[0][1]) #y만 나옴 
print(xy_train[0][0].shape) #(10, 200, 200, 1)
print(xy_train[0][1].shape) #(10,)
 
print(xy_train[1][0].shape) #(10, 200, 200, 1)
print(xy_train[1][1].shape) #(10,)
# print(xy_train[0][1].shape) 와 마찬가지로 batch_size만큼의 데이터가 들어가있음

print(xy_train[15][0].shape) #(10, 200, 200, 1)
print(xy_train[15][1].shape) #(10,)

# =>전체 160장의 이미지를 batch 10으로 짤랐으므로 행은 0부터 15까지 총 16개의 행이 있음.
# 이중에 0열은 x값 1열은 y값인 거
#######################################################################################################
#print(xy_train[16][0].shape) # 없음 ~
"""

print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'> => tuple는 list와 비슷하다. tuple은 한 번 생성하면 바꿀 수 없다.
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

#ImageDataGenerator은 이미지 데이터를 수치로 바꿔주고 x와 y를 numpy 형태의 데이터바꿔주고 이 x,y를 포함한 튜플형태의 tensorflow타입의 데이터로 바꿔줌

