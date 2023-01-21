from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. data
datasets = load_breast_cancer()
x= datasets['data']
y= datasets['target']
# print(x.shape,y.shape) #(569,30) (569,)
# print(datasets.DESCR)  #-> pandas 와 sklearn  DESCR 사용 방법 다름
# print(datasets.feature_names)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.85,shuffle=True,random_state=9)

#2. model
model =Sequential()
model.add(Dense(50, activation='linear',input_dim=30))
model.add(Dense(40,activation= 'relu'))
model.add(Dense(30,activation= 'relu'))
model.add(Dense(20,activation= 'relu'))
model.add(Dense(10,activation= 'relu'))
model.add(Dense(1,activation= 'sigmoid'))
        
#3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # loss function(손실 함수)  #https://curt-park.github.io/2018-09-19/loss-cross-entropy/
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode = 'min',patience=2, restore_best_weights=True, verbose = 1)

hist = model.fit(x_train,y_train, epochs=10, batch_size=3
          ,validation_split= 0.2, 
          verbose = 1, callbacks=[earlyStopping]
          )

#4.evaluate
# loss =model.evaluate(x_test,y_test)
# print('loss, accuracy : ' , loss)

loss,accuracy =model.evaluate(x_test,y_test)
print('loss : ' , loss)
print('accuracy :' , accuracy)
y_predict = model.predict(x_test)
y_predict = (y_predict >0.5)

print(y_predict)
print(y_test)

print(y_predict.shape)
print(y_test.shape)

# print(y_predict[:10])  # 실수형 수치 0~1
# print(y_test[:10]) # 0 또는 1 정수
#hint argu 과제



from sklearn.metrics import accuracy_score  # accuracy계 rmse,r2 이런 느낌
acc = accuracy_score(y_test, y_predict)   #= 오류 뜸 ValueError: Classification metrics can't handle a mix of binary and continuous targets // test 는 0또는 1인 정수형 수치임, predict 0~1 실수형 수치
print("accuracy_score : ", acc)






