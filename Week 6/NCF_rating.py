import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load data
ratings = pd.read_csv("D:/movies dataset/ml-32m/ratings.csv").head(10000)
movies = pd.read_csv("D:/movies dataset/ml-32m/movies.csv")

# Filter only movies that appear in ratings
movie_ids_used = ratings['movieId'].unique()
movies = movies[movies['movieId'].isin(movie_ids_used)]

# Create mappings
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user_id_map = {id: idx for idx, id in enumerate(user_ids)}
movie_id_map = {id: idx for idx, id in enumerate(movie_ids)}
movie_title_lookup = dict(zip(movies['movieId'], movies['title']))

# Map user/movie to numeric indices
ratings['user'] = ratings['userId'].map(user_id_map)
ratings['movie'] = ratings['movieId'].map(movie_id_map)

num_users = len(user_id_map)
num_movies = len(movie_id_map)

# Train-test split
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Build model
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))
user_embed = Embedding(num_users, 50)(user_input)
movie_embed = Embedding(num_movies, 50)(movie_input)
user_vec = Flatten()(user_embed)
movie_vec = Flatten()(movie_embed)

x = Concatenate()([user_vec, movie_vec])
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  # restrict between 0 and 1
output = Lambda(lambda y: y * 5)(x)     # scale to 0 to 5

model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(loss='mean_squared_error', optimizer=Adam(0.001))

# Train
model.fit(
    [train['user'], train['movie']], train['rating'],
    validation_split=0.1, epochs=5, batch_size=64, verbose=1
)

# Reverse mapping
movie_id_reverse_map = {v: k for k, v in movie_id_map.items()}

# Recommendation function
def ncf_recommend(user_id, top_n=5):
    if user_id not in user_id_map:
        print("User ID not found.")
        return []

    user_idx = user_id_map[user_id]
    watched_ids = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    watched_idx = [movie_id_map[mid] for mid in watched_ids if mid in movie_id_map]

    # Recommend only unwatched movies
    candidate_idxs = [i for i in range(num_movies) if i not in watched_idx]
    user_array = np.full(len(candidate_idxs), user_idx)

    preds = model.predict([user_array, np.array(candidate_idxs)], verbose=0).flatten()
    top_indices = preds.argsort()[-top_n:][::-1]

    top_movie_ids = [movie_id_reverse_map[i] for i in np.array(candidate_idxs)[top_indices]]
    top_ratings = preds[top_indices]

    # Return movie titles + predicted ratings
    return [(movie_title_lookup.get(mid, f"MovieID {mid}"), round(score, 2)) for mid, score in zip(top_movie_ids, top_ratings)]

# Precision & Recall evaluation
def precision_recall_at_k(k=5):
    hits = 0
    total_relevant = 0
    total_recommended = 0

    for uid in test['userId'].unique():
        if uid not in user_id_map:
            continue
        actual_movies = test[test['userId'] == uid]['movieId'].tolist()
        recs = ncf_recommend(uid, top_n=k)
        recommended_titles = [title for title, _ in recs]
        actual_titles = [movie_title_lookup.get(mid) for mid in actual_movies if mid in movie_title_lookup]

        hit_set = set(recommended_titles) & set(actual_titles)
        hits += len(hit_set)
        total_recommended += k
        total_relevant += len(actual_titles)

    precision = hits / total_recommended if total_recommended > 0 else 0
    recall = hits / total_relevant if total_relevant > 0 else 0
    return precision, recall

# Ask user
try:
    uid = int(input("Enter your User ID: "))
    how_many = int(input("How many movie recommendations do you want?: "))

    recommendations = ncf_recommend(uid, how_many)
    if recommendations:
        print("\nTop-N Recommendations (NCF):")
        for i, (title, pred) in enumerate(recommendations, 1):
            print(f"{i}. {title} (Predicted Rating: {pred})")

except ValueError:
    print("Enter a valid number for User ID ")
