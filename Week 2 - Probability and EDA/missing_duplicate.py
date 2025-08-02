import pandas as pd
import numpy as np
import seaborn as sns

#movies_csv
df = pd.read_csv("D:/movies dataset/ml-32m/movies.csv")

#rating_csv
dff= pd.read_csv("D:/movies dataset/ml-32m/ratings.csv")

#Finding if any duplicated data
print(df.duplicated())
print(df.duplicated().sum())
print(df.loc[df.duplicated()])

#Finding if any cell is empty
print(df.isna())
print(df.isna().sum())

df['genres'] = df['genres'].str.split('|')

#Reseting index if case index is not smooth
print(df.reset_index(drop=True))

