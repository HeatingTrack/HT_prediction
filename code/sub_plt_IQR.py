import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['font.family'] = 'Malgun Gothic'

data_21 = ['2101.csv', '2102.csv', '2103.csv', '2104.csv', '2105.csv', '2106.csv',
           '2107.csv', '2108.csv', '2109.csv', '2110.csv', '2111.csv', '2112.csv']
data_22 = ['2201.csv', '2202.csv', '2203.csv', '2204.csv', '2205.csv', '2206.csv',
           '2207.csv', '2208.csv', '2209.csv', '2210.csv', '2211.csv', '2212.csv']
data_23 = ['2301.csv', '2302.csv', '2303.csv', '2304.csv', '2305.csv', '2306.csv', '2307.csv', '2308.csv']


def draw_box_plots(year):
    num_files = 12
    num_cols = 4
    num_rows = (num_files - 1) // num_cols + 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    fig.suptitle('Box Plot of USEAMT')

    for i, one_mon in enumerate(year):
        data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/oneroom/oneroom_' + str(one_mon))
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        sns.boxplot(data=data['USEAMT'], orient='h', ax=ax)
        ax.set_title(one_mon.split('/')[-1])  # 파일 이름을 제목으로 설정

    plt.tight_layout()
    plt.show()


draw_box_plots(data_23)
