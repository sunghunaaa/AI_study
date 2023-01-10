from tensorflow.python.keras.models import Sequential
model = Sequential()
hist = model.fit(x,y,epochs=__,batch_size=32,
                 validation_split=0.2
                 verbose =1 
                 )


print(hist) #loss,val_loss 의 저장 위치가 나옴
print(hist.history) #loss, val_loss 값 다 나옴
print(hist.history['loss']) #loss값만
print(hist.history['val_loss']) #val_loss값만
