"""
Convolutional Neural Network 합성곱 신경망
인간의 시신경 구조를 모방한 기술
이미지를 인식하기 위헤 패턴을 찾는데 유용
데이터를 직접 학습하고 패턴을 사용해 이미지를 분류
자율주행자동차, 얼굴인식과 같은 객치 인식이나 computer vision이 필요한 분야에 많이 사용
이미지의 공간 정보를 유지한 채 학습을 하게 하는 모델(2D 그대로 작업함)
ex) 사람이 여러 데이터를 보고 기억한 후에 무엇인지 맞추는 것과 유사함
한 장의 컬러 사진은 3차원 데이터이다.
https://rubber-tree.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-CNN-Convolutional-Neural-Network-%EC%84%A4%EB%AA%85
"""



from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten



model = Sequential()
# 6만장의 사진이라면 input은 (60000,5,5,1)이라도 행무시 결론 input은 (5,5,1)  장수는 중요한 게 아님
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))   # (n,4,4,10)  n= 60000   
##(batch_shape, rows,columns,channels)  // channels 은 컬러 혹은 filters가 됨/batch_shape
model.add(Conv2D(filters=5, kernel_size=(2,2)))  #(n,3,3,5)
#########################################################


#위의 layer는 model.add(Conv2D(5,(2,2)))와 동일
#위의 layer는 model.add(Conv2D(5,2)))와 동일 2를 (2,2)로 인식

#########################################################
model.add(Flatten())   #(n,45)
model.add(Dense(units=10))   #(n,10)  
#input은 batch_size, input_dim   //  parameter = units
model.add(Dense(4, activation = 'relu'))    #지현 ,성환, 건률, 렐루(n,1)



model.summary()



# Input shape
# 4+D tensor with shape: batch_shape + (channels, rows, cols) if data_format='channels_first'
# 4+D tensor with shape: batch_shape + (rows, cols, channels) if data_format='channels_last'.


# Output shape
# 4+D tensor with shape: batch_shape + (filters, new_rows, new_cols) if data_format='channels_first' or 
# 4+D tensor with shape: batch_shape + (new_rows, new_cols, filters) if data_format='channels_last'. rows and cols values might have changed due to padding.




#   none = 데이터 갯수 !
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50   ------ CNN     

#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205       

#  flatten (Flatten)           (None, 45)                0    ------

#  dense (Dense)               (None, 10)                460(45 *10 + BIOS(1)*10 )  ------ DNN     

#  dense_1 (Dense)             (None, 1)                 11 (10 + 1*1)

# =================================================================
# Total params: 726
# Trainable params: 726
# Non-trainable params: 0
