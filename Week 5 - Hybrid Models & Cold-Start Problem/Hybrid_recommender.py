import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies= pd.read_csv("D:/movies dataset/ml-32m/movies.csv")
ratings= pd.read_csv("D:/movies dataset/ml-32m/ratings.csv")

movies= movies.head(10000)
ratings= ratings.head(10000)

movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_index = pd.Series(movies.index, index=movies['movieId'])

def content_based_recommend(movie_id, top_n=5):
    if movie_id not in movie_index:
        return []
    idx = movie_index[movie_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_ids = [movies.iloc[i[0]].movieId for i in sim_scores]
    return [movies[movies.movieId == mid]['title'].values[0] for mid in movie_ids]

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, _ = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

def collaborative_recommend(user_id, top_n=5):
    all_movies = movies['movieId'].unique()
    watched = ratings[ratings.userId == user_id]['movieId'].tolist()

    preds = []
    for movie in all_movies:
        if movie not in watched:
            rating = model.predict(user_id, movie).est
            preds.append((movie, rating))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [movies[movies.movieId == mid]['title'].values[0] for mid, _ in preds]

def hybrid_recommend(user_id, movie_id, top_n=5):
    content = content_based_recommend(movie_id, top_n * 2)
    collab = collaborative_recommend(user_id, top_n * 2)
    combined_titles = list(set(content + collab))[:top_n]
    return combined_titles

print("Hybrid Movie Recommender")

uid = int(input("Enter your User ID: "))
mid = int(input("Enter a Movie ID you like: "))
how_many = int(input("How many movie recommendations do you want?: "))

recs = hybrid_recommend(uid, mid, how_many)


for i, title in enumerate(recs, 1):
    print(f"{i}. {title}")
