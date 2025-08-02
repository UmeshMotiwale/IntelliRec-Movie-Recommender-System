import pandas as pd

movies= pd.read_csv("D:/movies dataset/ml-32m/movies.csv")
ratings= pd.read_csv("D:/movies dataset/ml-32m/ratings.csv")

# print(ratings.head())
# print(movies.head())

movies['genres'] = movies['genres'].str.split('|')
# print(movies.head())

combined = pd.merge(movies,ratings, on='movieId')
print (combined.tail(500000))