import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) # 그래프 크기
plt.plot(hist.history['loss'],c='red',marker='.',label='loss') #c= color ,marker= 선무늬
plt.plot(hist,history['val_loss'],c='blue',marker='.',label='val_loss')
plt.grid() # 모눈종이형태 그래프
plt.xlabel('epochs') # x축
plt.ylabel('loss') # y축
plt.title('_______') # 제목
plt.legend(loc='upper right') # loc = local // upper or down, right or left  조합에 따라 그래프 방향을 정해준다. /upper right 오른쪽윗방향그래프
# 그래프 위치는 자동오르 나오며 직접 지정도 가능하다.

plt.show()
