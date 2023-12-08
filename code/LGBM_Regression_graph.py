import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
    data = data[data['USEAMT'] >= 10]
    data_sample = data.sample(n=100, random_state=43)
    combined_data = pd.concat([combined_data, data_sample], ignore_index=True)

combined_data = combined_data.dropna(subset=['TEMP', 'SECT_1', 'SECT_2', 'SECT_3', 'SECT_4', 'SECT_5', 'SECT_6', 'SECT_7', 'SECT_8',
                             'SECT_9', 'SECT_10', 'SECT_11', 'H_DAY', 'T_STD'])
# 피처와 타겟 분리
X = combined_data[['TEMP', 'SECT_1', 'SECT_2', 'SECT_3', 'SECT_4', 'SECT_5', 'SECT_6', 'SECT_7', 'SECT_8', 'SECT_9',
                   'SECT_10', 'SECT_11', 'H_DAY', 'T_STD']]
Y = combined_data['USEAMT']

# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)


# LightGBM 모델 설정
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
}

# LightGBM 모델 학습
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=1000)

# LightGBM 모델 평가
y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
mse = mean_squared_error(y_valid, y_pred)
print(f'LightGBM 모델 - Root Mean Squared Error (MSE): {mse}')

# 실제 데이터 분포 그래프
plt.figure(figsize=(12, 6))
sns.lineplot(data=combined_data, x=combined_data.index, y='USEAMT', label='실제 데이터 분포', color='blue')

# LightGBM 모델 예측값 그래프
y_pred_line = pd.Series(data=y_pred, index=X_valid.index)
sns.lineplot(data=y_pred_line, label='LightGBM 모델 예측', color='red')

plt.xlabel('데이터 포인트')
plt.ylabel('USEAMT')
plt.title('실제 데이터 분포 및 LightGBM 모델 예측')
plt.legend()
plt.show()
