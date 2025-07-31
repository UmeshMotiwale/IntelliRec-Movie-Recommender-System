import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD

movies = pd.read_csv("D:/movies dataset/ml-32m/movies.csv")
ratings = pd.read_csv("D:/movies dataset/ml-32m/ratings.csv")

ratings = ratings.head(100000)  # optional

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

with open("svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

movie_dict = dict(zip(movies['movieId'], movies['title']))
with open("movie_dict.pkl", "wb") as f:
    pickle.dump(movie_dict, f)
