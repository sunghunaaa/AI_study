#1.data
import numpy as np
x1_datasets = np.array([range(100), range(301,401)]).transpose()
print(x1_datasets.shape) #(100,2) # 삼성전자 시가,고가


y1 = np.array(range(2001,2101)) # 삼성전자의 하루 뒤의 종가
print(y1.shape) #(100,)

y2 = np.array(range(201,301)) 
print(y2.shape) #(100,)

from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test, \
    y2_train,y2_test=train_test_split(x1_datasets,y1,y2,train_size=0.7,random_state=321)

print(x1_train.shape) #(70, 2)
print(y1_train.shape) #(70,)
print(y2_train.shape) #(70,)

print(x1_test.shape) #(30, 2)
print(y1_test.shape) #(30,)
print(y2_test.shape) #(30,) 


#2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate

#2-1. model1
input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

#2-5. model5 분기1
dense41 = Dense(41, activation='relu')(output1)
output4 = Dense(1)(dense41)

#2-6. model6 분기2
dense51 = Dense(51, activation='relu')(output1)
output5 = Dense(1)(dense51)


model = Model(inputs =[input1], outputs=[output4,output5])

model.summary()
#3. compile
model.compile(loss='mse', optimizer='adam')
model.fit(x1_train,[y1_train,y2_train],epochs=200,batch_size=32)

#4. predict
loss = model.evaluate(x1_test, [y1_test,y2_test])
y_pred1, y_pred2 = model.predict(x1_test)
print('loss :', loss) #loss가 3개인 이유
print('y_pred1 :', y_pred1)
print('y_pred2 :', y_pred2)
"""
loss : [1727.82666015625, 1373.2752685546875, 354.5513610839844]
y_pred1 : [[2130.186 ]
 [1985.5748]
 [2032.9884]
 [2002.1694]
 [1973.7212]
 [1940.5319]
 [1961.8678]
 [2120.7034]
 [1945.2731]
 [2016.3934]
 [2089.8845]
 [2040.1002]
 [2151.5222]
 [2025.8763]
 [2085.1433]
 [2101.7378]
 [2092.2551]
 [2030.6176]
 [2047.2123]
 [2051.9536]
 [2115.9622]
 [2011.6522]
 [2137.298 ]
 [1980.8333]
 [2106.4792]
 [2059.0657]
 [1978.4625]
 [1938.1608]
 [2009.2814]
 [1992.6866]]
y_pred2 : [[263.68292]
 [245.74492]
 [251.6262 ]
 [247.80338]
 [244.27454]
 [240.15765]
 [242.80426]
 [262.50662]
 [240.74582]
 [249.56775]
 [258.68378]
 [252.50839]
 [266.3295 ]
 [250.74405]
 [258.09564]
 [260.15408]
 [258.97784]
 [251.33214]
 [253.39061]
 [253.97876]
 [261.9185 ]
 [248.97964]
 [264.56506]
 [245.15678]
 [260.74222]
 [254.86095]
 [244.8627 ]
 [239.86356]
 [248.68556]
 [246.6271 ]]
"""

