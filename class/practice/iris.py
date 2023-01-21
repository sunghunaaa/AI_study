
from sklearn.datasets import load_iris
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. data
datasets = load_iris()
#print(datasets.DESCR)  # pandas  .describe() or .info()
# - class:
#                 - Iris-Setosa
#                 - Iris-Versicolour
#                 - Iris-Virginica
# y 컬럼 1개 class가 3개이지만 0,1,2로 3개일 뿐이지 컬럼은 1개임
# x 컬럼 4개 input_dim 4개
# Class Correlation 높을 수록 구분이 좋다 느낌

#print(datasets.feature_names) #pandas .columns
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
#pritnt(x) 열 확인
#pritnt(y) 확인
#print(x.shape,y.shape) # (150,4) (150,)
from keras.utils import to_categorical
y = to_categorical(y)
print(y)


x_train,x_test,y_train,y_test = train_test_split(x,y,
                        shuffle=True,  #False의 문제점은 y_test 전부 2나옴// 성능 개 쓰레기 됨// 즉, 하나의 동일한 값이 쏠리게 됨. 일 잘 못 해
                      random_state= True,
                      test_size= 0.2,
                      stratify=y)  # *** stratify y의 데이터가 분류형 데이터일 경에만 사용가능하다.



# *분류*에서 문제점 (회귀는 상관 없음)
# y_test 확인 해보면 2의 값이 50%를 모델은 데이터의 비율때문에 치우쳐져버림 따라서 데이터 비율이 중요함.
print(y_train)
print(y_test)

model = Sequential()
model.add(Dense(50, activation='relu', input_dim=4))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax')) # y columns 1개인데 output은 3임,



#compile
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=4, batch_size=1, validation_split=0.2, verbose =1)

loss, accuracy = model.evaluate(x_test, y_test)
#print ('loss : ', loss)
#print ('accuracy : ', accuracy)

# [[9.87073302e-01 1.27921505e-02 1.34546048e-04]  -> 실질 값 0
#  [2.85171042e-03 5.84370196e-01 4.12778169e-01]  -> 실질 값1
#  [3.64060998e-01 6.06894314e-01 2.90447641e-02]  -> 실질 값1
#  [7.34594930e-03 6.94499135e-01 2.98154920e-01]  -> 실질 값1
#  [3.08576785e-03 5.88103235e-01 4.08810973e-01]  -> 실질 값1 
#  [1.66782667e-03 4.96433169e-01 5.01899064e-01]  -> 실질 값2
#  [9.88901973e-01 1.09909466e-02 1.07210639e-04]  -> 실질 값0
#  [9.84876871e-01 1.49566596e-02 1.66521975e-04]  -> 실질 값0
#  [9.86874640e-01 1.29894735e-02 1.35868788e-04]  -> 실질 값0
#  [4.32070345e-02 8.41052711e-01 1.15740322e-01]]  -> 실질 값1

from sklearn.metrics import accuracy_score
import numpy as np
print(y_test.shape)  #(30,3)

y_predict = model.predict(x_test)  # 현재 y_predict 값은 0,1이 아닌 0.4239475238 이런식임
print(y_predict.shape) #(30,3)
y_predict = np.argmax(y_predict, axis =1 ) #위 문제점때문에 argmax 처리해준 거임
print(y_predict)#가로형태 [0 2 1 2 2 2 0 0 0 2 2 2 0 2 0 2 0 1 1 0 2 2 2 2 0 0 2 2 2 2]
print(y_predict.shape)  #(30,)



print(y_test) #세로형태 (30,3)
print(y_test.shape)
y_test = np.argmax(y_test, axis=1)
print(y_test) #가로형태
print(y_test.shape) #(30,)


acc = accuracy_score(y_test,y_predict)
print(acc)

#가로형태로 바꿔서 acc 비교해야 함. 당연한 거임


#argmax



