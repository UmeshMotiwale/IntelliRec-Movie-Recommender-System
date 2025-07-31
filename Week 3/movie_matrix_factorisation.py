import pandas as pd
import numpy as np

#ratings_csv by user
ratings = pd.read_csv("D:/movies dataset/ml-32m/ratings.csv").head(500)
ratings= ratings.drop(['timestamp'], axis=1)

# for getting unique users and movies and sorting
all_users = sorted(list(ratings['userId'].unique()))
all_movies = sorted(list(ratings['movieId'].unique()))

# Creating a map from userId/movieId to row and column index
user_to_index = {}
for i in range(len(all_users)):
    user_to_index[all_users[i]] = i

movie_to_index = {}
for j in range(len(all_movies)):
    movie_to_index[all_movies[j]] = j

#Creating the ratings matrix
num_users = len(all_users)
num_movies = len(all_movies)
M = np.zeros((num_users, num_movies))

# Filling the matrix
for row in ratings.itertuples():
    user = row.userId
    movie = row.movieId
    rating = row.rating
    i = user_to_index[user]
    j = movie_to_index[movie]
    M[i][j] = rating

#starting factorisation
N = 5  # Number of hidden features

P = np.random.rand(num_users, N)  # user features
Q = np.random.rand(num_movies, N)  # movie features

#Gradient decsent with error, improvements and steps
x = 0.001  
y = 0.001    
steps = 500    

#Matrix factorization using gradient descent
for step in range(steps):
    for i in range(num_users):
        for j in range(num_movies):
            if M[i][j] > 0: 
                prediction = np.dot(P[i], Q[j])
                error = M[i][j] - prediction
                for k in range(N):
                    P[i][k] = P[i][k] + x * (2 * error * Q[j][k] - y * P[i][k])
                    Q[j][k] = Q[j][k] + x * (2 * error * P[i][k] - y * Q[j][k])
    
    

#predictions
M_predicted = np.dot(P, Q.T)

# Show predicted ratings for any user 
user_id = 5
i = user_to_index[user_id]
predictions = M_predicted[i]

print("\nPredicted vs Actual Ratings for User 1:")
for j in range(num_movies):
    movie_id = all_movies[j]
    predicted_rating = round(predictions[j], 2)
    actual_rating = M[i][j]
    
    if actual_rating > 0:  # Show only if the user has actually rated it
        print(f"Movie ID: {movie_id}  Predicted: {predicted_rating}  Actual: {actual_rating}")

