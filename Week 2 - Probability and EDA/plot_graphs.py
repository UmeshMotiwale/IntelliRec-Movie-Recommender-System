import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

df = pd.read_csv("D:/movies dataset/ml-32m/movies.csv")
df['title'].value_counts().head(5).plot(kind='bar')
plt.title("Movie titles")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()