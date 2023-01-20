
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error,r2_score

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=13)
print(x_train.shape)



model = Sequential()
model.add(Dense(100,input_dim =10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=1,
          validation_split=0.2
          )


loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

r2 = r2_score(y_test,y_predict)

print('loss :', loss)
print('RMSE : ', RMSE(y_test,y_predict))
print('r2 : ', r2)
