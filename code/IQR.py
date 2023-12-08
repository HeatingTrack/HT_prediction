import numpy as np
import pandas as pd

data_21 = ['2101.csv', '2102.csv', '2103.csv', '2104.csv', '2105.csv', '2106.csv',
           '2107.csv', '2108.csv', '2109.csv', '2110.csv', '2111.csv', '2112.csv']
data_22 = ['2201.csv', '2202.csv', '2203.csv', '2204.csv', '2205.csv', '2206.csv',
           '2207.csv', '2208.csv', '2209.csv', '2210.csv', '2211.csv', '2212.csv']
data_23 = ['2301.csv', '2302.csv', '2303.csv', '2304.csv', '2305.csv', '2306.csv', '2307.csv', '2308.csv']


def IQR_filter(file_list):
    for mon in file_list:
        # 파일 불러오기
        data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/oneroom/oneroom_' + str(mon))

        # 원본 데이터 복사하여 새로운 데이터프레임 생성
        filtered_data = data.copy()

        # IQR 계산
        quartile_1, quartile_3 = np.percentile(data['USEAMT'], [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (1.5 * iqr)
        upper_bound = quartile_3 + (1.5 * iqr)

        # 'USEAMT' 열 값을 기준으로 필터링하여 모든 열 유지
        filtered_data = filtered_data.loc[
            (filtered_data['USEAMT'] >= lower_bound) & (filtered_data['USEAMT'] <= upper_bound)]

        # 필터링된 데이터를 CSV 파일로 저장
        filtered_data.to_csv('C:/Users/user/Desktop/Project/Gas/data/filtered/oneroom/oneroom_' + str(mon),
                             index=False, encoding='utf-8-sig')


IQR_filter(data_23)
