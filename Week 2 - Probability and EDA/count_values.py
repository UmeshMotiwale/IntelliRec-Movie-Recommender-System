import pandas as pd
import numpy as np
import seaborn as sns

#for viewing all the columns
pd.set_option('display.max_columns', 10)


#movies_csv
df = pd.read_csv("D:/movies dataset/ml-32m/movies.csv")

#rating_csv
dff= pd.read_csv("D:/movies dataset/ml-32m/ratings.csv")

combined = pd.merge(df,dff, on='movieId')
print(combined['genres'].value_counts())

print(combined['rating'].value_counts())
