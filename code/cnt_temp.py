import numpy as np
import pandas as pd

data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/22_temp.csv', encoding="cp949")
list_21 = [202101, 202102, 202103, 202104, 202105, 202106, 202107, 202108, 202109, 202110, 202111, 202112]
list_22 = [202201, 202202, 202203, 202204, 202205, 202206, 202207, 202208, 202209, 202210, 202211, 202212]

data['DAY'] = pd.to_datetime(data['DAY'], format='%Y%m')

for mon in list_22:
    filtered_data = data[(data['LOC'] == '양산시') & (data['DAY'].dt.strftime('%Y%m') == str(mon))]
    result_temp_values = filtered_data['TEMP']
    a = len(result_temp_values[result_temp_values <= -8])
    b = len(result_temp_values[result_temp_values <= -7])
    c = len(result_temp_values[result_temp_values <= -4])
    d = len(result_temp_values[result_temp_values <= -1])
    e = len(result_temp_values[result_temp_values <= 1])
    f = len(result_temp_values[result_temp_values <= 4])
    g = len(result_temp_values[result_temp_values <= 7])
    h = len(result_temp_values[result_temp_values <= 10])
    i = len(result_temp_values[result_temp_values <= 13])
    j = len(result_temp_values[result_temp_values <= 16])
    k = len(result_temp_values[result_temp_values <= 20])
    print(f"{mon}", a, b, c, d, e, f, g, h, i, j, k)

