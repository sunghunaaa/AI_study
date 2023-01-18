import numpy as np
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
###########################################################
from tensorflow.python.keras.layers import Dropout
###########################################################
path = './_save/'
#1.data
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test=train_test_split (x,y,test_size=0.3,shuffle=True,random_state=321)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#2.model
input1 = Input(shape=(13,))
dense1 = Dense(10,activation='relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(10,activation='linear')(drop1)
drop2 = Dropout(0.5)(dense2)
output1 = Dense(1,activation='linear')(drop2)
model = Model(inputs=input1, outputs=output1)
model.summary()
"""
0.5는 위 layer에서 0.5(절반)만큼 노드를 빼준다는 의미
0.2는 위 layer에서 1/5만큼 노드를 빼준다는 의미

<Dropout>
과적합을 방지하기 위해, 편향되지 않은 출력값을 내기 위해 0~1 사이의 확률로 중간 중간의 노드를 제거해주는 것
https://heytech.tistory.com/127
<Dropout 하는 방법>

-Sequential model
from tensorflow.keras.layers import Dropout
model.add(Dropout(0~1사이값:확률))

-Hamsu model
from tensorflow.keras.layers import Dropout
drop1 = Dropout(0~1사이값:확률))(dense1)
dense2 = Dense(40, activation='sigmoid')(drop1)

Dropout은 훈련할 때만 적용된
evaluate 평가할 때는 모든 데이터를 다 활용한다.
predict 예측할 때는 훈련해서 만들어진 함수, 가중치에 집어넣어서 값이 나오기 때문에 Dropout이 적용안된다.
"""
#3. compile,fit
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2,verbose=1)
#4. evaluate,predict
mse, mae = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

