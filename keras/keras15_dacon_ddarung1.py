import numpy as np
import pandas as pd #데이터 분석할 때 상당히 좋은 api (데이터 분석 쪽 sklearn 느낌)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data  #.현재 /하단
path = './_data/ddarung/'  #data의 위치 표시 (train) , path라는 이름으로 subnission, test, train 공통점 만들어 귀찮은 일 안 만듦
train_csv = pd.read_csv(path+'train.csv', index_col=0)  # train_csv이라는 변수
# train_csv = pd.read_csv('./_data/ddarung/train.csv', index_col=0) # index_col=0 처리 안 하면 id 열로 인식함
test_csv = pd.read_csv(path+'test.csv', index_col=0)
# print(train_csv)
# print(train_csv.shape) # (1459,10) 실질적 input_dim=9 count = y
submission = pd.read_csv(path + 'submission.csv', index_col=0)
# print(train_csv.columns)
# print(train_csv.info())  #결측치가 있는 데이터를 삭제해버리는 방법도 있다
# print(test_csv.info())  
# print(train_csv.describe())  
x = train_csv.drop(['count'], axis = 1)
# print(x) #(1459rows,9columns)
y = train_csv['count']
# print(y)
# print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7)

#model
model = Sequential()
model.add(Dense(100,input_dim=9))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#compile
model.compile(loss = 'mse' ,optimizer = 'adam')
model.fit(x_train,y_train,epochs=10,batch_size=32)

#predict
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  # test_csv 넣으면 안 됨 ,머신을 러닝 시키고 나중에 test를 넣어 submission으로 시험 제출하는 거임 
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


#제출할 놈 
y_submit = model.predict(test_csv)
