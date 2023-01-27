import numpy as np
x1_datasets = np.array([range(100), range(301,401)]).transpose()
print(x1_datasets.shape) #(100,2) # 삼성전자 시가,고가

x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).T
print(x2_datasets.shape) #(100,3) # 아모레 시가, 고가, 종가

y = np.array(range(2001,2101)) # 삼성전자의 하루 뒤의 종가
print(y.shape) #(100,)

##### 변수 3개 이상 넣기 가능
from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(x1_datasets,x2_datasets,y,train_size=0.7,random_state=321)
"""
print(x1_train.shape)
print(x2_train.shape)
print(y_train.shape)
print(x1_test.shape)
print(x2_test.shape)
print(y_test.shape)

(70, 2)
(70, 3)
(70,)
(30, 2)
(30, 3)
(30,)
"""
#2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. model1
input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

#2-2. model2
input2 = Input(shape=(3,))
dense21 = Dense(21, activation='linear', name='ds21')(input2)
dense22 = Dense(22, activation='linear', name='ds22')(dense21)
output2 = Dense(23, activation='linear', name='ds23')(dense22)

#2-3. model_merge 모델 병합
from tensorflow.keras.layers import concatenate # concatenate : 사슬처럼 엮다. (단순하게) 붙이다.

merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(10, activation='relu', name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs =[input1, input2], outputs=last_output )

model.summary()

"""
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 2)]          0           []

 ds11 (Dense)                   (None, 11)           33          ['input_1[0][0]']

 input_2 (InputLayer)           [(None, 3)]          0           []

 ds12 (Dense)                   (None, 12)           144         ['ds11[0][0]']

 ds21 (Dense)                   (None, 21)           84          ['input_2[0][0]']

 ds13 (Dense)                   (None, 13)           169         ['ds12[0][0]']

 ds22 (Dense)                   (None, 22)           484         ['ds21[0][0]']

 ds14 (Dense)                   (None, 14)           196         ['ds13[0][0]']

 ds23 (Dense)                   (None, 23)           529         ['ds22[0][0]']

 mg1 (Concatenate)              (None, 37)           0           ['ds14[0][0]',
                                                                  'ds23[0][0]']

 mg2 (Dense)                    (None, 10)           380         ['mg1[0][0]']

 mg3 (Dense)                    (None, 10)           110         ['mg2[0][0]']

 last (Dense)                   (None, 1)            11          ['mg3[0][0]']

==================================================================================================
"""
#compile
model.compile(loss='mse' , optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=8)

#evaluate
loss = model.evaluate([x1_test,x2_test],y_test)
print ('loss : ', loss)



