#modeling 내부 구성 확인할 때 사용
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. data
x = np.array([1,2,3])
y = np.array([1,2,3])


#2. model
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary() 

# #Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense (Dense)                (None, 5)                 10        
# _________________________________________________________________
# dense_1 (Dense)              (None, 4)                 24        
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 15        
# _________________________________________________________________
# dense_3 (Dense)              (None, 2)                 8         
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 3         
# =================================================================
# Total params: 60
# Trainable params: 60
# Non-trainable params: 0
# _________________________________________________________________
# 각 단계에서 input은 1을 더해서 곱하는 계산을 한다. (bios)
