import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

ratings = pd.read_csv("D:/movies dataset/ml-32m/ratings.csv").head(10000)
movies = pd.read_csv("D:/movies dataset/ml-32m/movies.csv").head(10000)

user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()

user_id_map = {id: idx for idx, id in enumerate(user_ids)}
movie_id_map = {id: idx for idx, id in enumerate(movie_ids)}

ratings['user'] = ratings['userId'].map(user_id_map)
ratings['movie'] = ratings['movieId'].map(movie_id_map)

num_users = len(user_id_map)
num_movies = len(movie_id_map)

train, test = train_test_split(ratings, test_size=0.2, random_state=42)
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))
user_embed = Embedding(num_users, 50)(user_input)
movie_embed = Embedding(num_movies, 50)(movie_input)
user_vec = Flatten()(user_embed)
movie_vec = Flatten()(movie_embed)

concat = Concatenate()([user_vec, movie_vec])
x = Dense(128, activation='relu')(concat)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dense(1)(x)

model = Model(inputs=[user_input, movie_input], outputs=x)
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))


history = model.fit(
    [train['user'], train['movie']], train['rating'],
    validation_split=0.1,
    epochs=5, batch_size=64, verbose=1
)

movie_title_lookup = dict(zip(movies['movieId'], movies['title']))

def ncf_recommend(user_id, top_n=5):
    if user_id not in user_id_map:
        return []

    user_idx = user_id_map[user_id]
    watched = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    watched_idx = [movie_id_map[mid] for mid in watched if mid in movie_id_map]

    all_movie_indices = np.array([i for i in range(num_movies) if i not in watched_idx])
    user_array = np.full_like(all_movie_indices, user_idx)

    preds = model.predict([user_array, all_movie_indices], verbose=0).flatten()
    top_indices = preds.argsort()[-top_n:][::-1]

    movie_id_reverse_map = {v: k for k, v in movie_id_map.items()}
    top_movie_ids = [movie_id_reverse_map[i] for i in all_movie_indices[top_indices]]
    return [movie_title_lookup[mid] for mid in top_movie_ids if mid in movie_title_lookup]

uid = int(input("Enter your User ID: "))
how_many = int(input("How many movie recommendations do you want?: "))

ncf_recs = ncf_recommend(uid, how_many)
print("\nTop-N Recommendations (NCF):")
for i, title in enumerate(ncf_recs, 1):
    print(f"{i}. {title}")
