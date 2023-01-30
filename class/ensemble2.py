#1.data
import numpy as np
x1_datasets = np.array([range(100), range(301,401)]).transpose()
print(x1_datasets.shape) #(100,2) # 삼성전자 시가,고가

x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).T
print(x2_datasets.shape) #(100,3) # 아모레 시가, 고가, 종가

x3_datasets = np.array([range(100,200), range(1301, 1401)]).T
print(x3_datasets.shape) #(100,2)

y = np.array(range(2001,2101)) # 삼성전자의 하루 뒤의 종가
print(y.shape) #(100,)

from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,x3_train,x3_test,y_train,y_test=train_test_split(x1_datasets,x2_datasets,x3_datasets,y,train_size=0.7,random_state=321)

print(x1_train.shape) #(100, 2)
print(x2_train.shape) #(100, 3)
print(x3_train.shape) #(100, 2)
print(y_train.shape) #(100,)


print(x1_test.shape) #(70, 2)
print(x2_test.shape) #(70, 3)
print(x3_test.shape) #(70, 2)
print(y_test.shape) #(70,)

#2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate

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

#2-3. model3
input3 = Input(shape=(2,))
dense31 = Dense(31, activation='relu', name='ds31')(input3)
dense32 = Dense(32, activation='relu', name='ds32')(dense31)
output3 = Dense(33, activation='relu', name='ds33')(dense32)

#2-3. model_merge 모델 병합
merge1 = concatenate([output1, output2, output3], name='mg1')
merge2 = Dense(10, activation='relu', name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs =[input1, input2, input3], outputs=last_output )

#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train,x2_train,x3_train],y_train,epochs=200,batch_size=32)

#4. predict
loss = model.evaluate([x1_test,x2_test,x3_test],y_test)
y_pred = model.predict([x1_test,x2_test,x3_test])
print('loss :', loss)
print('y_pred :', y_pred)
"""
loss : 76.15544128417969
y_pred : [[2098.8972]
 [2017.5983]
 [2044.2535]
 [2026.9274]
 [2010.9346]
 [1992.2753]
 [2004.2706]
 [2093.5662]
 [1994.9412]
 [2034.9242]
 [2076.2402]
 [2048.252 ]
 [2110.8403]
 [2040.2552]
 [2073.5745]
 [2082.9043]
 [2077.5732]
 [2042.9208]
 [2052.2502]
 [2054.9158]
 [2090.9004]
 [2032.2589]
 [2102.8953]
 [2014.9326]
 [2085.5698]
 [2058.9143]
 [2013.6   ]
 [1990.9427]
 [2030.926 ]
 [2021.5966]] 
"""
