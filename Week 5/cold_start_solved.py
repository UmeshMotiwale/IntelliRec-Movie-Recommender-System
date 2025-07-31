import pandas as pd
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

def is_new_user(user_id):
    return user_id not in ratings['userId'].values

def is_new_movie(movie_id):
    return movie_id not in movies['movieId'].values

def hybrid_recommend(user_id, movie_id, top_n=5):
    new_user = is_new_user(user_id)
    new_movie = is_new_movie(movie_id)
    
    if new_user and not new_movie:
        print("New user detected. Using content-based recommendations.")
        return content_based_recommend(movie_id, top_n)
    
    elif not new_user and new_movie:
        print("New movie detected. Using collaborative recommendations.")
        return collaborative_recommend(user_id, top_n)
    
    elif new_user and new_movie:
        print(" New user and new movie. Using content-based recommendations.")
        # Recommend most similar movies to a random one or based on genres
        return movies['title'].sample(top_n).tolist()
    
    else:
        print("Known user and movie. Using hybrid recommendation.")
        content = content_based_recommend(movie_id, top_n * 2)
        collab = collaborative_recommend(user_id, top_n * 2)
        combined = list(set(content + collab))[:top_n]
        return combined
    
def precision_recall_at_k(user_id, k=5, threshold=4.0):
    # Get all movies rated by user in test set
    test_user_ratings = ratings[ratings.userId == user_id]

    if test_user_ratings.empty:
        print("No historical data to evaluate Precision/Recall for this user.")
        return 0, 0

    # Relevant items = movies rated >= threshold
    relevant_items = set(test_user_ratings[test_user_ratings['rating'] >= threshold]['movieId'].tolist())
    if not relevant_items:
        print("User has no relevant (highly-rated) items to evaluate against.")
        return 0, 0

    # Predict top-K movies for the user (using hybrid, without asking for movie_id)
    recommended_titles = hybrid_recommend(user_id, 1, k)  # 1 = dummy movie id

    # Get movieId from titles
    recommended_ids = movies[movies['title'].isin(recommended_titles)]['movieId'].tolist()
    
    # True positives = relevant and recommended
    true_positives = set(recommended_ids) & relevant_items

    precision = len(true_positives) / k
    recall = len(true_positives) / len(relevant_items)
    return precision, recall
    

print("Cold-Start Aware Hybrid Recommender")
try:
    uid = int(input("Enter your User ID: "))
    mid = int(input("Enter a Movie ID you like: "))
    how_many = int(input("How many movie recommendations do you want?: "))
    
    recs = hybrid_recommend(uid, mid, how_many)
    print("\n Recommendations for You:")
    for i, title in enumerate(recs, 1):
           print(f"{i}. {title}")



except Exception as e:
    print("Error:", e)
