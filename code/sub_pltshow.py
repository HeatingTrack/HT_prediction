import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Malgun Gothic'


def draw_graph(k, year):
    selected_column = 'USEAMT'
    fig, axs = plt.subplots(3, 4, figsize=(20, 20))
    fig.suptitle(f'{selected_column}월별 사용량')

    for i, one_mon in enumerate(year):
        data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/' + k + '/' + k + '_' + str(one_mon))
        x_values = range(len(data['USEAMT']))
        y_values = data[selected_column].values  # y 축 값
        row = i // 4
        col = i % 4
        ax = axs[row, col]
        ax.plot(x_values, y_values)
        ax.set_title(f'{one_mon}')
        ax.set_xlabel('index')
        ax.set_ylabel('USEAMT')
        ax.set_ylim(0, 2500)
    for i in range(12, 12):
        fig.delaxes(axs.flatten()[i])
    plt.tight_layout()
    plt.show()


data_21 = ['2101.csv', '2102.csv', '2103.csv', '2104.csv', '2105.csv', '2106.csv',
           '2107.csv', '2108.csv', '2109.csv', '2110.csv', '2111.csv', '2112.csv']
data_22 = ['2201.csv', '2202.csv', '2203.csv', '2204.csv', '2205.csv', '2206.csv',
           '2207.csv', '2208.csv', '2209.csv', '2210.csv', '2211.csv', '2212.csv']
data_23 = ['2301.csv', '2302.csv', '2303.csv', '2304.csv', '2305.csv', '2306.csv', '2307.csv', '2308.csv']

draw_graph('oneroom', data_23)
