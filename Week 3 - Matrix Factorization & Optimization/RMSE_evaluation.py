#Let’s check the model accuracy, by seeing how close its predictions are to real ratings."
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

ratings_csv = "D:/movies dataset/ml-32m/ratings.csv"
movies_csv = "D:/movies dataset/ml-32m/movies.csv"

#Load limited rows for performance
ratings = pd.read_csv(ratings_csv, nrows=500000)
movies = pd.read_csv(movies_csv)

#Convert IDs to strings
ratings['userId'] = ratings['userId'].astype(str)
ratings['movieId'] = ratings['movieId'].astype(str)
movies['movieId'] = movies['movieId'].astype(str)


unique_users = ratings['userId'].nunique()
unique_movies = ratings['movieId'].nunique()
print(f"Loaded {len(ratings)} ratings from {unique_users} unique users and {unique_movies} unique movies.\n")


reader = Reader(rating_scale=(0.0, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


algo = SVD()

#Evaluate using 5-fold cross-validation
print("Evaluating SVD model using 3-fold cross-validation: ")
results = cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=True)

#to show average RMSE
average_rmse = round(results['test_rmse'].mean(), 4)

print(f"\nAverage RMSE across 5 folds: {average_rmse}")
