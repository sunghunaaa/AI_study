import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


#1. data
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) # (10,)
# y = ?? y가 없어
x= np.array([[1,2,3],   
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9]])
y= np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7,3) ,(7,)

x = x.reshape(7,3,1)
"""
            ([[[1],[2],[3]],   
             [[2],[3],[4]],
             [[3],[4],[5]],
             [[4],[5],[6]],
             [[5],[6],[7]],
             [[6],[7],[8]],
             [[7],[8],[9]])
"""
print(x.shape) 

#2. model
model = Sequential()
model.add(SimpleRNN(256, input_shape=(3,1)))
"""
연산 방법 
[[1],[2],[3]]
[1] 일때 [2]야 
[2] 일때 [3]이야
[3] 일때 y값이야 
이런 식임
실질적으로 [3] -> h1 -> y값(4) 중요한 부분
앞부분은 무의미해보이나, 버리고싶지 않은 데이터임// ex) 주식에서 버릴 수 없음, 날씨데이터 등등..
"""
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3. compile, fit
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=2000,batch_size=1, verbose=2)


#4. predict
loss = model.evaluate(x,y)
print(loss)
y_pred = np.array([8,9,10]).reshape(1,3,1)
#y_pred = np.array([[[8],[9],[10]]])
result = model.predict(y_pred)
print(result)
"""
0.051102638244628906
[[10.936562]]
"""
