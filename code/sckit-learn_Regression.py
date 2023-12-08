import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

plt.rcParams['font.family'] = 'Malgun Gothic'

# 데이터 파일 목록
data_21 = ['2101.csv', '2102.csv', '2103.csv', '2104.csv', '2105.csv', '2106.csv',
           '2107.csv', '2108.csv', '2109.csv', '2110.csv', '2111.csv', '2112.csv']
data_22 = ['2201.csv', '2202.csv', '2203.csv', '2204.csv', '2205.csv', '2206.csv',
           '2207.csv', '2208.csv', '2209.csv', '2210.csv', '2211.csv', '2212.csv']
data_23 = ['2301.csv', '2302.csv', '2303.csv', '2304.csv', '2305.csv', '2306.csv', '2307.csv', '2308.csv']

combined_data_list = data_21 + data_22

# 모든 데이터를 하나의 DataFrame에 병합
combined_data = pd.DataFrame()

for mon in combined_data_list:
    data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/filtered/oneroom/oneroom_' + str(mon), encoding="cp949")
    data = data[data['USEAMT'] != 0]
    data_sample = data.sample(n=15000, random_state=43)
    combined_data = pd.concat([combined_data, data_sample], ignore_index=True)

# 피처와 타겟 분리
X = combined_data[['TEMP', 'SECT_1', 'SECT_2', 'SECT_3', 'SECT_4', 'SECT_5', 'SECT_6', 'SECT_7', 'SECT_8', 'SECT_9', 'H_DAY']]
Y = combined_data['USEAMT']

# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)

# 선형 회귀 모델 초기화
model = LinearRegression()

# 모델 학습
model.fit(X_train, y_train)

# 모델 예측
y_pred = model.predict(X_valid)

mse = mean_squared_error(y_valid, y_pred)
print(f'선형 회귀 모델 - Mean Squared Error (MSE): {mse}')

# 실제 데이터 분포 그래프
plt.figure(figsize=(12, 6))
sns.lineplot(data=combined_data, x=combined_data.index, y='USEAMT', label='실제 데이터 분포', color='blue')

# 모델 예측값 그래프
y_pred_line = pd.Series(data=y_pred, index=X_valid.index)
sns.lineplot(data=y_pred_line, label='선형 회귀 모델 예측', color='red')

plt.xlabel('데이터 포인트')
plt.ylabel('USEAMT')
plt.title('실제 데이터 분포 및 선형 회귀 모델 예측')
plt.legend()
plt.show()
