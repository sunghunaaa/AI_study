import pandas as pd #csv data 읽고 쓸 때 사용하는 중요한 api이다.

#1. data
path = './_data/ddarung/' 
# .은 현재 열려있는 폴더 중 가장 상위 파일이다 
# /은 하위 폴더로 이동을 의미한다. 
# .과 /을 사용하여 csv파일이 위치한 곳으로 경로를 설정해주면 된다.
# 현재 csv파일의 위치는 study 파일 -> _data 파일 -> ddarung 파일에 존재하고 .은 study 파일이 된다.
# 위 ./_data/ddarung/ 는 자주 쓰일 문구임으로 효율을 위해 path로 정의해둔다.

train_csv = pd.read_csv(path+'train.csv', index_col=0)
#'index_col=0' 은 받은 test.csv data의 번호 매김(id)을 위해 쓰였고, 실제 column 중 특징이 되지 못 하는 data임으로 훈련에서 제거를 위해 사용.
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission = pd.read_csv(path+'submission.csv', index_col=0)

# index_col=0 안 하면 id생김 무조건 해야 됨

print(train_csv) 
# 훈련에 필요한 column값을 알아낸다. input_dim이 된다


x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']
# 훈련에 사용될 x와 y는 주어진 훈련용 csv(train_csv)에서 얻는다.
# x는 id열과 count열을 제외한 나머지 값이 된다. 따라서 남은 'count'는 위 source로 제거한다. 
# axis = 1은 열을 뜻하고, axis = 0 은 행을 뜻한다.
# y는 count열이 된다. 따라서 위 source로 얻는다.

from sklearn.models_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,shuffle=True,random_state=79)

#model

#compile, fit

#predict
y_predict = model.predict(x_test)

#RMSE
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
  return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)

#sumit
y_sumit = model.predict(test_csv)
# 제출해야하는 y값은 test용 csv(test_csv)가 x값이 된다.

#submission completed
submission['count'] = y_submit
print(submission)
# 제출될 submission data 확인한다.

submission.to_csv(path + 'submission_0105.csv')
# path 경로를 지난 폴더에 새로 완성된 submission_0105.csv 파일로 추출한다.
