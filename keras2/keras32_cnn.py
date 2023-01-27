from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
# 6만장의 사진이라면 input은 (60000,5,5,1)이라도 행무시 결론 input은 (5,5,1)  장수는 중요한 게 아님
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))   # (n,4,4,10)  n= 60000   
##(batch_shape, rows,columns,channels)  // channels 은 컬러 혹은 filters가 됨/batch_shape
model.add(Conv2D(filters=5, kernel_size=(2,2)))  #(n,3,3,5)
# 윗 모델과 같은 거임 model.add(Conv2D(5,(2,2)))
# 윗 모델과 같은 거임 model.add(Conv2D(5,2))

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
# _________________________________________________________________

