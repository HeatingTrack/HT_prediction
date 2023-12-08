import numpy as np
import pandas as pd

data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/23_temp.csv', encoding="cp949")
list_20 = [202011, 202012]
list_21 = [202101, 202102, 202103, 202104, 202105, 202106, 202107, 202108, 202109, 202110, 202111, 202112]
list_22 = [202201, 202202, 202203, 202204, 202205, 202206, 202207, 202208, 202209, 202210, 202211, 202212]
list_23 = [202301, 202302, 202303, 202304, 202305, 202306, 202307, 202308]
data['DAY'] = pd.to_datetime(data['DAY'], format='%Y%m')

# for mon in list_20:
#     filtered_data = data[(data['DAY'].dt.strftime('%Y%m') == str(mon))]
#     result_temp_values = filtered_data['TEMP']
#     std_deviation = np.std(result_temp_values)
#     print(f"{mon} 데이터의 표준편차:", std_deviation)


for mon in list_23:
    filtered_data = data[(data['LOC'] == '울산') & (data['DAY'].dt.strftime('%Y%m') == str(mon))]
    result_temp_values = filtered_data['TEMP']
    std_deviation = np.std(result_temp_values)
    print(f"{mon} 데이터의 표준편차:", std_deviation)