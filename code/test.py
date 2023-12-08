import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/oneroom/oneroom_2302.csv', encoding='cp949')
x_values = range(len(data['USEAMT']))
y_values = data['USEAMT'].values

plt.plot(x_values, y_values)
plt.show()
