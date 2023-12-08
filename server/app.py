from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

pre_temp = [-2.8, 2.1, 8, 13, 15.9, 21.8, 26.2, 24.3, 20.9, 14, 6.9, 0.2,
            -2.7, -1.4, 6.9, 13.1, 17.4, 22.5, 25.8, 24.7, 20.6, 12.9, 8, -3.9,
            -2.2, 1.2, 8.6, 12.9, 18, 22.6, 25.4, 26.2, 22.2, 13.7, 7.0, -1.1]
pre_sect = [[5, 8, 14, 20, 20, 26, 30, 31, 31, 31, 31],
            [0, 1, 4, 9, 9, 18, 25, 27, 28, 28, 28, 28],
            [0, 0, 0, 0, 0, 2, 12, 22, 30, 31, 31, 31],
            [0, 0, 0, 0, 0, 0, 0, 4, 17, 24, 29],
            [0, 0, 0, 0, 0, 0, 0, 0, 5, 16, 27],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 1, 11, 16, 17, 26],
            [0, 0, 0, 0, 1, 6, 16, 23, 30, 30],
            [2, 3, 5, 9, 17, 27, 31, 31, 31, 31, 31],
            [0, 2, 13, 24, 27, 31, 31, 31, 31, 31, 31],
            [0, 0, 6, 16, 21, 27, 18, 31, 31, 31, 31],
            [0, 0, 0, 0, 0, 7, 16, 25, 29, 31, 31, 31],
            [0, 0, 0, 0, 0, 0, 1, 6, 16, 22, 31],
            [0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 28],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 13],
            [0, 0, 0, 0, 0, 0, 1, 6, 24, 30],
            [0, 0, 0, 1, 1, 3, 9, 24, 30, 30],
            [6, 8, 15, 22, 27, 31, 31, 31, 31, 31, 31],
            [3, 3, 8, 19, 26, 29, 30, 31, 31, 31, 31],
            [0, 0, 0, 6, 13, 25, 28, 28, 28, 28, 28],
            [0, 0, 0, 0, 0, 3, 12, 19, 28, 31, 31],
            [0, 0, 0, 0, 0, 0, 0, 7, 14, 27, 30],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 25],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 1, 12, 26, 31],
            [0, 0, 0, 0, 2, 7, 17, 24, 27, 31, 0],
            [2, 3, 8, 13, 21, 29, 31, 31, 31, 31, 31]
            ]
pre_std = [5.910487337091136, 4.5886450358799955, 2.654244828334585, 3.1012183985581467, 2.9362848960267383, 1.6634669285027046, 1.8262794326381173, 1.935306443480437, 1.0282455392020375, 5.358799545629537, 3.283730940392177, 4.104706569899241,
           2.788119585322759, 3.0430646307528164, 3.268093793749662, 3.564954417660904, 2.648479429454081, 3.070150195384946, 1.6873712229739806, 2.8529773410661425, 3.274812971758845, 3.5852339094156362, 3.19775789508142, 3.9375272326164614,
           4.297142287827043, 2.1753371777116977, 3.5949641040585236, 3.139357683773333, 2.6263849723322785, 1.8132843130629022, 1.4905529775825574, 2.0406891166367367, 2.253049784921171, 1.9484885681759019, 4.026390719021469, 3.7619238109809445]
pre_hday = [11, 10, 9, 8, 12, 8, 9, 10, 11, 12, 8, 8,
            11, 10, 10, 9, 10, 10, 10, 9, 10, 12, 8, 9,
            11, 8, 8, 10, 10, 9, 10, 9, 11, 12, 8, 11]
pre_year = [2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112,
            2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212,
            2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312
            ]
columns_to_keep = ['USEAMT', 'TEMP', 'SECT_1', 'SECT_2', 'SECT_3', 'SECT_4', 'SECT_5', 'SECT_6', 'SECT_7', 'SECT_8', 'SECT_9', 'SECT_10', 'SECT_11', 'H_DAY', 'T_STD']


# 전체 데이터 파일 목록
data_21 = ['2101.csv', '2102.csv', '2103.csv', '2104.csv', '2105.csv', '2106.csv',
           '2107.csv', '2108.csv', '2109.csv', '2110.csv', '2111.csv', '2112.csv']
data_22 = ['2201.csv', '2202.csv', '2203.csv', '2204.csv', '2205.csv', '2206.csv',
           '2207.csv', '2208.csv', '2209.csv', '2210.csv', '2211.csv', '2212.csv']
data_23 = ['2301.csv', '2302.csv', '2303.csv', '2304.csv', '2305.csv', '2306.csv', '2307.csv', '2308.csv']

combined_data_list = data_21 + data_22 + data_23

# 모든 데이터를 하나의 DataFrame에 병합
combined_data = pd.DataFrame()


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


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return "hello world"


@app.route('/predict', methods=['POST'])
def prediction():
    global combined_data
    pre_data = request.json
    print(pre_data['pre_month'])
    for mon in combined_data_list:
        data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/filtered/oneroom/oneroom_' + str(mon), usecols=columns_to_keep)
        data = data[data['USEAMT'] >= 10]
        data_sample = data.sample(n=100, random_state=43)
        mon = mon.replace(".csv", "")
        for x in range(len(pre_data['user_input'])):
            if str(pre_data['user_input'][x]['year']) == mon:
                pre_df = pd.DataFrame(pre_sect[pre_data['pre_month']]).T.add_prefix('SECT_')
                pre_df.columns = 'SECT_' + (pre_df.columns.str.extract('(\d+)').astype(int) + 1).astype(str)
                pre_df = pre_df.rename(columns=lambda x: x[0])
                pre_df.insert(0, 'USEAMT', pre_data['user_input'][x]['use'])
                pre_df['H_DAY'] = pre_hday[pre_year.index(pre_data['user_input'][x]['year'])]
                pre_df['T_STD'] = pre_std[pre_year.index(pre_data['user_input'][x]['year'])]
                print(pre_df)
                data_sample = pd.concat([data_sample, pre_df], axis=1, ignore_index=True)
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

    idx = int(pre_data['pre_month'])
    predict = predict_gas_usage(model, pre_temp[idx], *pre_sect[idx], pre_hday[idx], pre_std[idx])
    prediction = {'prediction': round(predict, 1)}
    print(prediction)
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(host='192.168.0.247', port=5000, debug=True)
