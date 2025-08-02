import pandas as pd
import numpy as np


df = pd.read_csv("D:/movies dataset/ml-32m/movies.csv")

#To delete a column
df = df.drop(['genres'], axis=1)
print(df)

#To delete a row
df = df.drop([3])
print(df)

#To change a column dtype
#df['Title']= pd.to_numeric(df['Title'])
#practically not possible since is a string but way remains the same

#To rename a column
df = df.rename(columns={'movieId' : 'MovieId'})
df = df.rename(columns= {'title' : 'Title'})

print(df)