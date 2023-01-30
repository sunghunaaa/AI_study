#1.data
import numpy as np
x1_datasets = np.array([range(100), range(301,401)]).transpose()
print(x1_datasets.shape) #(100,2) # 삼성전자 시가,고가

x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).T
print(x2_datasets.shape) #(100,3) # 아모레 시가, 고가, 종가

x3_datasets = np.array([range(100,200), range(1301, 1401)]).T
print(x3_datasets.shape) #(100,2)

y1 = np.array(range(2001,2101)) # 삼성전자의 하루 뒤의 종가
print(y1.shape) #(100,)

y2 = np.array(range(201,301)) 
print(y2.shape) #(100,)

from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test, \
    x3_train,x3_test,y1_train,y1_test, \
    y2_train,y2_test=train_test_split(x1_datasets,x2_datasets,x3_datasets,y1,y2,train_size=0.7,random_state=321)

print(x1_train.shape) #(70, 2)
print(x2_train.shape) #(70, 3)
print(x3_train.shape) #(70, 2)
print(y1_train.shape) #(70,)
print(y2_train.shape) #(70,)

print(x1_test.shape) #(30, 2)
print(x2_test.shape) #(30, 3)
print(x3_test.shape) #(30, 2)
print(y1_test.shape) #(30,)
print(y2_test.shape) #(30,) 


#2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate,Concatenate

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

#2-4. model4_merge 모델 병합
merge1 = concatenate([output1, output2, output3],name='merge1')
merge2 = Dense(10, activation='relu', name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

#2-5. model5 분기1
dense41 = Dense(41, activation='relu')(last_output)
output4 = Dense(1)(dense41)

#2-6. model6 분기2
dense51 = Dense(51, activation='relu')(last_output)
output5 = Dense(1)(dense51)


model = Model(inputs =[input1, input2, input3], outputs=[output4,output5])

model.summary()
#3. compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train,x2_train,x3_train],[y1_train,y2_train],epochs=200,batch_size=32)

#4. predict
loss = model.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])
loss1 = model.evaluate([x1_test,x2_test,x3_test],y1_test)
loss2 = model.evaluate([x1_test,x2_test,x3_test],y2_test)
y_pred1, y_pred2 = model.predict([x1_test,x2_test,x3_test])
print('loss :', loss) #loss가 3개인 이유
print('loss1 :', loss1)
print('loss2 :', loss2)
print('y_pred1 :', y_pred1)
print('y_pred2 :', y_pred2)


