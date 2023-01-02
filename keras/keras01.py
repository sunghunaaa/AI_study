import tensorflow as tf
print(tf.__version__)
import numpy as np

#1. 데이터
x = np.array([1,2,3,])
y = np.array([1,2,3,])

#2. 모델구성
from tensorflow.keras.models import Sequential  # tensorflow에 있는 keras에 있는 models에서 Sequential을 가져와라
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))  # (output, input) //Dense - 한덩어리 x= np.array([1,2,3]) , y = np.array([1,2,3])

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') #mae= mean absolute error  - 컴파일
model.fit(x, y, epochs=2000)   #fit - 훈련, epochs - 훈련을 몇 번 시킬건지

#4. 평가,예측
result = model.predict([4])
print('결과 : ', result)

# 줄 copy - 커서 깜박이는 중 ctrl + c 하면 줄 복사 됨


# - 주석 
 


