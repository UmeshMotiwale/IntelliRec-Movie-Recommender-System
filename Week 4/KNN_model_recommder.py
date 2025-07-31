import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

#inserting dataset
ratings_csv = "D:/movies dataset/ml-32m/ratings.csv"
movies_csv = "D:/movies dataset/ml-32m/movies.csv"

ratings = pd.read_csv(ratings_csv, nrows=500000)  # Limit for speed
movies = pd.read_csv(movies_csv)

num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()

print(f"🎬 Loaded {len(ratings)} ratings from {num_users} users and {num_movies} movies.\n")

#Converting to strings
ratings['userId'] = ratings['userId'].astype(str)
ratings['movieId'] = ratings['movieId'].astype(str)
movies['movieId'] = movies['movieId'].astype(str)


reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

#Train KNN model

sim_options = {
    'name': 'cosine',  # similarity metric
    'user_based': True  # set to False for item-based
}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

#Lets take input from the user!!
user_input = input(f" Enter a userId (1 to {num_users}): ").strip()
movie_input = input(" Enter a movieId (you can check in movies.csv): ").strip()

# Check if the IDs exist
if user_input in ratings['userId'].values and movie_input in ratings['movieId'].values:
    prediction = algo.predict(user_input, movie_input)

   
    movie_title = movies[movies['movieId'] == movie_input]['title'].values
    movie_title = movie_title[0] if len(movie_title) > 0 else f"Movie ID {movie_input}"

    print(f"\n Prediction: User {user_input} is expected to rate \"{movie_title}\" around {round(prediction.est, 2)} stars.")
else:
    print(" Either the user ID or movie ID you entered is not found in the dataset.")
