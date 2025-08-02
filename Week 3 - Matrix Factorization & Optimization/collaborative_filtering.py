import pandas as pd
from surprise import SVD, Dataset, Reader

ratings_csv = "D:/movies dataset/ml-32m/ratings.csv"
movies_csv = "D:/movies dataset/ml-32m/movies.csv"

# Loading specific nrows for memory efficiency
ratings = pd.read_csv(ratings_csv, nrows=500000) 
movies = pd.read_csv(movies_csv)

#Converting IDs to strings
ratings = ratings.copy()
ratings['userId'] = ratings['userId'].astype(str)
ratings['movieId'] = ratings['movieId'].astype(str)
movies['movieId'] = movies['movieId'].astype(str)

unique_users = ratings['userId'].nunique()
unique_movies = ratings['movieId'].nunique()
print(f" Loaded {len(ratings)} ratings from {unique_users} unique users and {unique_movies} unique movies.\n")

reader = Reader(rating_scale=(0.0, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

print(" Training SVD model...")
algo = SVD()
algo.fit(trainset)

#Asking user for input
user_id = input(f"Enter a userId (from 1 to {unique_users}): ").strip()
movie_id = input("Enter a movieId: ").strip()

#In case of validate inputs
if user_id not in ratings['userId'].unique():
    print(f" ERROR! User ID {user_id} not found in dataset.")
elif movie_id not in ratings['movieId'].unique():
    print(f"ERROR! Movie ID {movie_id} not found in dataset.")
else:
    
    title_row = movies[movies['movieId'] == movie_id]['title']
    movie_title = title_row.values[0] if not title_row.empty else f"Movie ID {movie_id}"

    #Predict rating
    prediction = algo.predict(user_id, movie_id)
    print(f"\n Predicted rating of user {user_id} for movie \"{movie_title}\": {round(prediction.est, 2)}")
