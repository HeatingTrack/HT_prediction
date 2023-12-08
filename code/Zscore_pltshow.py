import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('C:/Users/user/Desktop/Project/Gas/data/filtered/oneroom/oneroom_2101.csv')
plt.figure(figsize=(8, 6))
sns.boxplot(data=data['USEAMT'], orient="h")
plt.xlabel("Value")
plt.title("Box Plot of Data")
plt.show()
