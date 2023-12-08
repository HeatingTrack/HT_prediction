import pandas as pd

data_21 = ['2101.csv', '2102.csv', '2103.csv', '2104.csv', '2105.csv', '2106.csv',
           '2107.csv', '2108.csv', '2109.csv', '2110.csv', '2111.csv', '2112.csv']
data_22 = ['2201.csv', '2202.csv', '2203.csv', '2204.csv', '2205.csv', '2206.csv',
           '2207.csv', '2208.csv', '2209.csv', '2210.csv', '2211.csv', '2212.csv']
data_23 = ['2301.csv', '2302.csv', '2303.csv', '2304.csv', '2305.csv', '2306.csv', '2307.csv', '2308.csv']

for mon in data_23:
    data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/normal/' + str(mon), encoding='cp949')
    oneroom = data[data['HSHD_TYPE_NM'] == '다가구주택(원룸)']
    oneroom_df = pd.DataFrame(oneroom)
    oneroom_df.to_csv('C:/Users/user/Desktop/Project/Gas/data/oneroom/oneroom_' + str(mon), index=False,
                      encoding='utf-8-sig')
    billa = data[data['HSHD_TYPE_NM'] == '다세대주택(빌라)']
    billa_df = pd.DataFrame(billa)
    billa_df.to_csv('C:/Users/user/Desktop/Project/Gas/data/billa/billa_' + str(mon), index=False, encoding='utf-8-sig')
    apt = data[data['HSHD_TYPE_NM'] == '연립/아파트(공동주택)']
    apt_df = pd.DataFrame(apt)
    apt_df.to_csv('C:/Users/user/Desktop/Project/Gas/data/apt/apt_' + str(mon), index=False, encoding='utf-8-sig')


# for mon in data_22:
#     data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/normal/' + str(mon), encoding='cp949')
#     oneroom = data[data['HSHD_TYPE_NM'] == '다가구주택(원룸)']
#     oneroom_df = pd.DataFrame(oneroom)
#     oneroom_df.to_csv('C:/Users/user/Desktop/Project/Gas/data/oneroom/oneroom_' + str(mon), index=False,
#                       encoding='utf-8-sig')
#     billa = data[data['HSHD_TYPE_NM'] == '다세대주택(빌라)']
#     billa_df = pd.DataFrame(billa)
#     billa_df.to_csv('C:/Users/user/Desktop/Project/Gas/data/billa/billa_' + str(mon), index=False, encoding='utf-8-sig')
#     apt = data[data['HSHD_TYPE_NM'] == '연립/아파트(공동주택)']
#     apt_df = pd.DataFrame(apt)
#     apt_df.to_csv('C:/Users/user/Desktop/Project/Gas/data/apt/apt_' + str(mon), index=False, encoding='utf-8-sig')
#
#
# for mon in data_23:
#     data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/normal/' + str(mon), encoding='cp949')
#     oneroom = data[data['HSHD_TYPE_NM'] == '다가구주택(원룸)']
#     oneroom_df = pd.DataFrame(oneroom)
#     oneroom_df.to_csv('C:/Users/user/Desktop/Project/Gas/data/oneroom/oneroom_' + str(mon), index=False,
#                       encoding='utf-8-sig')
#     billa = data[data['HSHD_TYPE_NM'] == '다세대주택(빌라)']
#     billa_df = pd.DataFrame(billa)
#     billa_df.to_csv('C:/Users/user/Desktop/Project/Gas/data/billa/billa_' + str(mon), index=False, encoding='utf-8-sig')
#     apt = data[data['HSHD_TYPE_NM'] == '연립/아파트(공동주택)']
#     apt_df = pd.DataFrame(apt)
#     apt_df.to_csv('C:/Users/user/Desktop/Project/Gas/data/apt/apt_' + str(mon), index=False, encoding='utf-8-sig')
#

