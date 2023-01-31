#https://www.kaggle.com/competitions/dogs-vs-cats
#ReduceLROnPlateau : learning rate 가장 많이 쓰임
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
path = 'C:/_data/hong/'
train_datagen = ImageDataGenerator(
    rescale=1./255. ,
)
 
hong = train_datagen.flow_from_directory(
    path,    
    target_size=(150,150),
    batch_size= 100000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)
# test = train_datagen.flow_from_directory(
#    'C:/_data/dogcat/test1/',    
#     target_size=(150,150),
#     batch_size= 100000,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True,
# )

print(hong[0][1].shape)
print(hong[0][0].shape)

np.save('C:/_data/hong.npy', arr=hong[0][0]) 
# np.save('C:/_data/dogcat_num/x_train.npy', arr=xy_train[0][0]) 
# np.save('C:/_data/dogcat_num/y_train.npy', arr=xy_train[0][1]) 
# np.save('C:/_data/dogcat_num/x_test.npy', arr=test[0][0])



