import pandas as pd
import numpy as np

df = pd.read_csv("D:/movies dataset/ml-32m/movies.csv")


print(df.shape)
print(df.columns)
print(df.head())
print(df.dtypes)
print(df.describe())


