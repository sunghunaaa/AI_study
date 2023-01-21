#[과제, 실습]
# R^2 0.62 이상
from sklearn.datasets import load_diabetes
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(datasets.feature_names) 
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets.DESCR) 
# - age     age in years
# - sex
# - bmi     body mass index
# - bp      average blood pressure
# - s1      tc, total serum cholesterol
# - s2      ldl, low-density lipoproteins
# - s3      hdl, high-density lipoproteins
# - s4      tch, total cholesterol / HDL
# - s5      ltg, possibly log of serum triglycerides level
# - s6      glu, blood sugar level
# print(x)
# print(x.shape)  #(442, 10)
# print(y)
# print(y.shape)  #(442,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=79)

model = Sequential()
model.add(Dense(500,input_dim=10))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=500,batch_size=1)

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  
print("RMSE : ", RMSE(y_test,y_predict))
  
r2= r2_score(y_test,y_predict)
print("R2 : ", r2)


# loss :  2575.45947265625
# RMSE :  50.74898304683466
# R2 :  0.6230478386413124 
