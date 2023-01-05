from sklearn.datasets import load_diabetes  # load_  / load_뒤에 원하는 주제 입력 

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(datasets.feature_names) 
print(datasets.DESCR)

print(x.shape) # 열확인하여 dim값 확인한다.
print(y.shape) # x의 행과 일치하는 확인, output dim 값 확인한다.
