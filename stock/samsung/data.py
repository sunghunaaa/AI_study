import pandas as pd
sf1 = pd.read_csv("./_data/stock/samsung.csv", index_col=0, header=None, encoding='cp949', sep=',')
af1 = pd.read_csv("./_data/stock/amore.csv", index_col=0, header=None, encoding='cp949', sep=',')
# print(sf1.info) # [1980 rows x 16 columns]
# print(af1.info) # [2221 rows x 16 columns] # 헤드에 index붙음
# 결측치 삭제
sf1 = sf1.dropna()
af1 = af1.dropna()
#pandas데이터에서 열 삭제
sf1 = sf1.drop(af1.columns[[4,5,6,9,10,11,12,13,14,15]],axis=1)

for i in range(len(sf1.index)):
    for j in range(len(sf1.iloc[i])):
        sf1.iloc[i,j] = int(sf1.iloc[i,j].replace(',',''))
print(sf1)  # , 아주 잘 없어짐
print(sf1.shape) #(1977, 6)

sf_sub = sf1[:5]
print('여기',sf_sub)

af1 = af1.drop(af1.columns[[4,5,6,9,10,11,12,13,14,15]],axis=1)
af1 = af1[:1977] # 행 갯수 맞춰주

for i in range(len(af1.index)):
    for j in range(len(af1.iloc[i])):
        af1.iloc[i,j] = int(af1.iloc[i,j].replace(',',''))
print(af1.shape) #(1977, 6)
print(af1) # , 아주 잘 없어짐

af_sub = af1[:5]
print('여기',af_sub)




sf1 = sf1.values
af1 = af1.values
sf_sub = sf_sub.values
af_sub = af_sub.values
print(sf_sub.shape)
print(af_sub.shape)

print(type(sf1), type(af1))  #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

import numpy as np
np.save('./_data/stock/sam.npy', arr=sf1)
np.save('./_data/stock/amor.npy', arr=af1)
np.save('./_data/stock/sam_sub.npy', arr=sf_sub)
np.save('./_data/stock/amor_sub.npy', arr=af_sub)

