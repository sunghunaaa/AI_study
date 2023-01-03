
#sklearn import
from sklearn.model_selection import train_test_split

#sklearn
x_train, x_test, y_train, y_test = train_test_split(
           x,y,
           train_size = 0.7, 
           test_size = 0.3,
           shuffle = true,
           random_state=123
           ) 

#train, test 위치 중요.
#train_size, test_size 둘 중 하나만 작성해도 됨.
#shuffle 작성 x시 True가 기본값임
#random_state 작성 x시 계속해서 랜덤하게 뽑기 됨. random_state에 난수 입력시 해당 난수는 고정이라 랜덤하게 뽑힌 숫자가 똑같이 계속 나온다.
           
