import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'

# 데이터 파일 목록
data_21 = ['2101.csv', '2102.csv', '2103.csv', '2104.csv', '2105.csv', '2106.csv',
           '2107.csv', '2108.csv', '2109.csv', '2110.csv', '2111.csv', '2112.csv']
data_22 = ['2201.csv', '2202.csv', '2203.csv', '2204.csv', '2205.csv', '2206.csv',
           '2207.csv', '2208.csv', '2209.csv', '2210.csv', '2211.csv', '2212.csv']
data_23 = ['2301.csv', '2302.csv', '2303.csv', '2304.csv', '2305.csv', '2306.csv', '2307.csv', '2308.csv']

combined_data_list = data_21 + data_22 + data_23

# 모든 데이터를 하나의 DataFrame에 병합
combined_data = pd.DataFrame()

for mon in combined_data_list:
    data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/filtered/oneroom/oneroom_' + str(mon))
    data = data.dropna(subset=['TEMP', 'SECT_1', 'SECT_2', 'SECT_3', 'SECT_4', 'SECT_5', 'SECT_6', 'SECT_7', 'SECT_8',
                               'SECT_9', 'SECT_10', 'SECT_11', 'H_DAY', 'T_STD'])
    data = data[data['USEAMT'] >= 10]

    data_sample = data.sample(n=100, random_state=43)
    combined_data = pd.concat([combined_data, data_sample], ignore_index=True)

# 피처와 타겟 분리
X = combined_data[['TEMP', 'SECT_1', 'SECT_2', 'SECT_3', 'SECT_4', 'SECT_5', 'SECT_6', 'SECT_7', 'SECT_8', 'SECT_9',
                   'SECT_10', 'SECT_11', 'H_DAY', 'T_STD']]
Y = combined_data['USEAMT']

# 데이터 스케일 조정 (Min-Max 스케일링)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# 신경망 모델 설정
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
model.fit(X_train, y_train, epochs=100, batch_size=32)

# 모델 평가
y_pred = model.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
print(f'신경망 모델 - Root Mean Squared Error (MSE): {mse}')

# 실제값과 예측값을 데이터프레임으로 만들기
valid_df = pd.DataFrame({'Real': y_valid, 'Prediction': y_pred.flatten()}, index=np.arange(len(y_valid)))

# 그래프 그리기
plt.figure(figsize=(12, 6))
sns.lineplot(data=valid_df['Real'], label='실제 데이터 분포', color='blue')
sns.lineplot(data=valid_df['Prediction'], label='신경망 모델 예측', color='red')

plt.xlabel('데이터 포인트')
plt.ylabel('USEAMT')
plt.title('실제 데이터 분포 및 신경망 모델 예측')
plt.legend()
plt.show()
