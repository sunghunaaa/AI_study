"""
데이터 정제, 가공 -> 성능 매우 좋아짐   참고) w 초기 값은 random, bios 초기 값은 0

ex) 
x = ( 1, 2, 100000, 4, 7, 2500000, 3000000, 500000000, 4300000000)
y = ( 37, 23, 5, 150, 630, 11, 7, 8, 1300)

y = wx if) w값이 300만이라하면 연산오류로 bomb 됨// 또, 큰 숫자의 경우 메모리에 계속 저장되어 연산 장애를 일으켜 방해한다.
computer architecture는 부동소숫점 연산에 매우 좋다. -> 어떤 작은 수를 가져오더라도 연산 오류 bomb이 없다.

따라서 x의 값을 줄일 필요가 있음. y값은 그대로 고정해도 무관하다.

scale 에는 minmaxscaler, standardscaler 두 종류 배웠음
"""
""" 
#1. minmaxscaler
최소값:0, 최대값:1로 만든다.
과정과 공식은 x_scl = (x - x_min) / (x_max - x_min) 으로 해주면 됨
"""
"""
#2. standardscaler
평균에 근접한 수가 가장 많은 정규분포표로 만든다.
 - 한쪽으로 치우쳐진 정규분포표는 좋지 못 한 데이터이다.
치우쳐져 있을 경우 표준화 작업이 필요하다. 
numpy : z = (x- mean())/std()   mean : 평균 std : 편차 
"""

#3. minmaxscaler 추가
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) # 기준은 train 따라서 값은(0~1)이다. // test는 음수, 1보다 큰 수 갖을 수 있다. 값의 크기는 위 공식에 넣으면 나온다.
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# 만약 submission 파일이 있다면 test csv파일도 scaling을 해주고 결과 값을 도출해야 함.
 



