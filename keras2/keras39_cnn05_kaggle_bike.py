
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd



##1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})                     

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
y = train_csv['count']   # output
# print(x.shape, y.shape)   # (10886, 8) (10886,)
# print(train_csv.describe())
# print(train_csv.info)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.8,
    random_state=341)

x = df.values

print(y_train.shape, y_test.shape) 
#(8708,) (2178,)
print(x_train.shape, x_test.shape) 
#(8708, 8) (2178, 8)

x_train = x_train.reshape(8708,8,1,1)
x_test= x_test.reshape(2178,8,1,1)
print(x_train.shape, x_test.shape) 


#2. model

model = Sequential()
model.add(Conv2D(64, kernel_size=(2,1), input_shape=(8,1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.summary()

##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=30, batch_size=1,
          validation_split=0.125,  verbose=1)

##4. 평가, 예측

print("================== 1. 기본 출력 =================")
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse, ' / mae:', mae)

y_predict = model.predict(x_test)


# from sklearn.metrics import mean_squared_error, r2_score

# r2 = r2_score(y_test, y_predict)
# print("R2:", r2)

"""

"""