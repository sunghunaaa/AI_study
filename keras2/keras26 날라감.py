from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)  # - 기준은 x_train이 됨
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# submission 자료가 있다면 test.csv 파일도 scaler를 해줘야 함.