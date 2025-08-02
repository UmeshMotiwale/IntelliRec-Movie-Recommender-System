import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("D:/movies dataset/ml-32m/movies.csv")
ratings = pd.read_csv("D:/movies dataset/ml-32m/ratings.csv")

# Limit to 10,000 for speed
movies = movies.head(10000)
ratings = ratings.head(10000)

# Preprocess genres for content-based filtering
movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_index = pd.Series(movies.index, index=movies['movieId'])

# Content-based recommendation function
def content_based_recommend(movie_id, top_n=5):
    if movie_id not in movie_index:
        return []
    idx = movie_index[movie_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_ids = [movies.iloc[i[0]].movieId for i in sim_scores]
    return [(movies[movies.movieId == mid]['title'].values[0], sim_scores[i][1]) for i, mid in enumerate(movie_ids)]

# Train collaborative filtering model
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
model = SVD()
model.fit(trainset)

# Collaborative filtering recommendation
def collaborative_recommend(user_id, top_n=5):
    all_movies = movies['movieId'].unique()
    watched = ratings[ratings.userId == user_id]['movieId'].tolist()

    preds = []
    for movie in all_movies:
        if movie not in watched:
            pred = model.predict(user_id, movie)
            preds.append((movie, pred.est))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    return [(movies[movies.movieId == mid]['title'].values[0], round(score, 2)) for mid, score in preds]

# Utility functions
def is_new_user(user_id):
    return user_id not in ratings['userId'].values

# Hybrid recommender that needs only userId
def hybrid_recommend(user_id, top_n=5):
    if is_new_user(user_id):
        print("🧊 Cold-start: New user detected. Using content-based recommendations.")
        # Pick a random popular movie the user hasn't seen
        top_movie = ratings['movieId'].value_counts().index[0]
        recs = content_based_recommend(top_movie, top_n)
        return [(title, round(score * 5, 2)) for title, score in recs]  # Convert sim score to 0–5 scale
    else:
        print("🧠 Known user. Using hybrid recommendation.")
        collab = collaborative_recommend(user_id, top_n * 2)
        # Use top-N similar genres from the user's most watched genres
        user_movies = ratings[ratings.userId == user_id]['movieId'].tolist()
        top_movie = user_movies[0] if user_movies else movies['movieId'].iloc[0]
        content = content_based_recommend(top_movie, top_n * 2)

        # Merge & deduplicate
        seen = set()
        combined = []
        for title, score in collab + content:
            if title not in seen:
                combined.append((title, score))
                seen.add(title)
            if len(combined) == top_n:
                break
        return combined

# --- Main interaction ---
print("🎬 Hybrid Recommender System (Cold-Start Aware)")
try:
    uid = int(input("Enter your User ID: "))
    how_many = int(input("How many movie recommendations do you want?: "))

    recs = hybrid_recommend(uid, how_many)
    print("\n📽️  Recommended Movies with Predicted Ratings:")
    for i, (title, score) in enumerate(recs, 1):
        print(f"{i}. {title}  (Predicted Rating: {score})")

except Exception as e:
    print("❌ Error:", e)
