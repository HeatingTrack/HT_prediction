import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

# 전체 데이터 파일 목록
data_21 = ['2101.csv', '2102.csv', '2103.csv', '2104.csv', '2105.csv', '2106.csv',
           '2107.csv', '2108.csv', '2109.csv', '2110.csv', '2111.csv', '2112.csv']
data_22 = ['2201.csv', '2202.csv', '2203.csv', '2204.csv', '2205.csv', '2206.csv',
           '2207.csv', '2208.csv', '2209.csv', '2210.csv', '2211.csv', '2212.csv']
data_23 = ['2301.csv', '2302.csv', '2303.csv', '2304.csv', '2305.csv', '2306.csv', '2307.csv', '2308.csv']

combined_data_list = data_21 + data_22

# 모든 데이터를 하나의 DataFrame에 병합
combined_data = pd.DataFrame()

for mon in combined_data_list:
    data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/filtered/oneroom/oneroom_' + str(mon))
    data = data[data['USEAMT'] >= 10]
    data_sample = data.sample(n=10000, random_state=43)
    combined_data = pd.concat([combined_data, data_sample], ignore_index=True)

combined_data = combined_data.dropna(
    subset=['TEMP', 'SECT_1', 'SECT_2', 'SECT_3', 'SECT_4', 'SECT_5', 'SECT_6', 'SECT_7', 'SECT_8',
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


def predict_gas_usage(model, temperature, sect_1, sect_2, sect_3, sect_4, sect_5, sect_6, sect_7, sect_8, sect_9,
                      sect_10, sect_11, h_day, t_std):
    # 입력된 변수들을 DataFrame 형태로 만들기
    data_to_predict = pd.DataFrame({
        'TEMP': [temperature],
        'SECT_1': [sect_1],
        'SECT_2': [sect_2],
        'SECT_3': [sect_3],
        'SECT_4': [sect_4],
        'SECT_5': [sect_5],
        'SECT_6': [sect_6],
        'SECT_7': [sect_7],
        'SECT_8': [sect_8],
        'SECT_9': [sect_9],
        'SECT_10': [sect_10],
        'SECT_11': [sect_11],
        'H_DAY': [h_day],
        'T_STD': [t_std]
    })

    # 모델을 사용하여 예측
    predicted_usage = model.predict(data_to_predict)[0]
    return predicted_usage


temperature_input = -2.2
sectors_input = [3, 3, 8, 18, 26, 29, 30, 31, 31, 31, 31]  # 각 섹터의 데이터 입력
h_day_input = 11
t_std_input = 11.228652841196007

# 예측 함수 호출
predicted_result = predict_gas_usage(model, temperature_input, *sectors_input, h_day_input, t_std_input)
print("Predicted Gas Usage:", round(predicted_result, 1))

# 학습된 모델을 lgbm_model.pkl로 내보내기
import joblib
joblib.dump(model, '../server/lgbm_model.pkl')
