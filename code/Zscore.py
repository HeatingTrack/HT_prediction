import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_21 = ['2101.csv', '2102.csv', '2103.csv', '2104.csv', '2105.csv', '2106.csv',
           '2107.csv', '2108.csv', '2109.csv', '2110.csv', '2111.csv', '2112.csv']
data_22 = ['2201.csv', '2202.csv', '2203.csv', '2204.csv', '2205.csv', '2206.csv',
           '2207.csv', '2208.csv', '2209.csv', '2210.csv', '2211.csv', '2212.csv']
data_23 = ['2301.csv', '2302.csv', '2303.csv', '2304.csv', '2305.csv', '2306.csv', '2307.csv', '2308.csv']


def Z_Score_filter(list):
    for mon in list:
        data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/oneroom/oneroom_' + str(mon))
        mean = data['USEAMT'].mean()
        std = data['USEAMT'].std()
        z_scores = np.abs((data['USEAMT'] - mean) / std)
        threshold = 3
        filtered_data = data[z_scores <= threshold]
        filtered_data.to_csv('C:/Users/user/Desktop/Project/Gas/data/filtered/oneroom/oneroom_' + str(mon),
                             index=False, encoding='utf-8-sig')


Z_Score_filter(data_21)
