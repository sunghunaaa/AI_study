import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime
es = EarlyStopping(monitor='val_loss',mode='min',patience=200,restore_best_weights=True,verbose=1)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/'
filename = '{epoch:05d}-{val_loss:.4f}.hdf5' 
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only=True, filepath= filepath+'k30_'+date+'_'+filename)


sf1 = np.load('./_data/stock/sam.npy',allow_pickle=True) 
af1 = np.load('./_data/stock/amor.npy',allow_pickle=True)
# print(sf1, sf1.shape) #(1977, 6)
# print(af1, af1.shape) #(1977, 6)
sf_sub = np.load('./_data/stock/sam_sub.npy',allow_pickle=True)
af_sub = np.load('./_data/stock/amor_sub.npy',allow_pickle=True)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:]
        tmp_y = dataset[x_end_number:y_end_number, 1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy5(sf1, 5, 1)
x2, y2 = split_xy5(af1, 5, 1)
# print (x1.shape) # (1972, 5, 6)
# print (x2.shape) # (1972, 5, 6)
# print(y1, y1.shape) #(1972, 1) 

from sklearn.model_selection import train_test_split # cross_val_score
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,random_state=30,train_size=0.7)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2,random_state=30,train_size=0.7)
# print (x1_train.shape,x1_test.shape) # (1380, 5, 6) (592, 5, 6)
# print (y1_train.shape,y1_test.shape) # (1380, 1) (592, 1)
# print (x2_train.shape,x2_test.shape) # (1380, 5, 6) (592, 5, 6)
# print (y2_train.shape,y2_test.shape) # (1380, 1) (592, 1)

x1_train =x1_train.reshape(1380,30)
x1_test =x1_test.reshape(592,30)
x2_train =x2_train.reshape(1380,30)
x2_test =x2_test.reshape(592,30)
x1_sub = sf_sub.reshape(1,30)
x2_sub = af_sub.reshape(1,30)

from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler() 
scaler2 = StandardScaler()
scaler1.fit(x1_train)
scaler2.fit(x2_train)

x1_train_s = scaler1.transform(x1_train)
x1_test_s = scaler1.transform(x1_test)
x2_train_s = scaler2.transform(x2_train)
x2_test_s = scaler2.transform(x2_test)
x1_sub = scaler1.transform(x1_sub)
x2_sub = scaler2.transform(x2_sub)

x1_train =x1_train.reshape(1380,30,1)
x1_test =x1_test.reshape(592,30,1)
x2_train =x2_train.reshape(1380,30,1)
x2_test =x2_test.reshape(592,30,1)
x1_sub = x1_sub.reshape(1,30,1)
x2_sub = x2_sub.reshape(1,30,1)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Conv1D


input1 = Input(shape=(30,1))
dense1 = Conv1D(60,30,activation='relu')(input1)
dense2 = Dense(30,activation='relu')(dense1)
output1 = Dense(15,activation='linear')(dense2)

input2 = Input(shape=(30,1))
dense21 = Conv1D(60,30,activation='relu')(input2)
dense22 = Dense(30,activation='relu')(dense21)
output2 = Dense(15,activation='linear')(dense22)

merge = concatenate([output1,output2])
merge1 = Dense(30,activation='relu')(merge)
merge2 = Dense(15,activation='relu')(merge1)
output3 = Dense(1,activation='linear')(merge2)

y1_train = y1_train.astype(float)
y1_test = y1_test.astype(float)
y2_train = y2_train.astype(float)
y2_test = y2_test.astype(float)

model = Model(inputs=[input1, input2], outputs = output3)
model.compile(loss='mae', optimizer='adam')

hist = model.fit([x1_train_s, x2_train_s], y1_train,epochs=130,batch_size=5,validation_split=0.30,verbose=4 ,callbacks=[es,mcp])

loss = model.evaluate([x1_train_s, x2_train_s], y1_train)
print('loss : ', loss)
print("====================================")
print('val_loss : ', hist.history['val_loss'])  
print("====================================")


y_pred = model.predict([x1_sub,x2_sub])

print('주가:' , y_pred)

#주가: [[[60774.312]]]



