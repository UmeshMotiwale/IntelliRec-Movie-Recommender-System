import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from surprise import Dataset, Reader, KNNBasic
from collections import defaultdict

ratings_path = "D:/movies dataset/ml-32m/ratings.csv"
movies_path = "D:/movies dataset/ml-32m/movies.csv"

ratings_df = pd.read_csv(ratings_path, nrows=100000) 
movies_df = pd.read_csv(movies_path)

# Convert ratings to binary: liked (1) if rating >= 4, else 0
ratings_df['liked'] = (ratings_df['rating'] >= 4).astype(int)

ratings_df['userId'] = ratings_df['userId'].astype('category')
ratings_df['movieId'] = ratings_df['movieId'].astype('category')
user_id_map = dict(enumerate(ratings_df['userId'].cat.categories))
movie_id_map = dict(enumerate(ratings_df['movieId'].cat.categories))

ratings_df['userId_code'] = ratings_df['userId'].cat.codes
ratings_df['movieId_code'] = ratings_df['movieId'].cat.codes

X = ratings_df[['userId_code', 'movieId_code']]
y = ratings_df['liked']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

reader = Reader(rating_scale=(0.5, 5.0))
surprise_data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = surprise_data.build_full_trainset()
knn_algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
knn_algo.fit(trainset)

#Precision@5 and Recall@5
def get_top_k_predictions(predictions, k=5):
    top_k = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_k[uid].append((iid, est))
    for uid in top_k:
        top_k[uid] = sorted(top_k[uid], key=lambda x: x[1], reverse=True)[:k]
    return top_k

def precision_recall_at_k(top_k_preds, actual_likes, k=5):
    precisions = []
    recalls = []
    for uid in actual_likes:
        pred_iids = set(iid for iid, _ in top_k_preds.get(uid, []))
        true_iids = set(actual_likes[uid])
        num_hit = len(pred_iids & true_iids)

        precision = num_hit / k
        recall = num_hit / len(true_iids) if true_iids else 0

        precisions.append(precision)
        recalls.append(recall)
    return np.mean(precisions), np.mean(recalls)

# Gather actual liked movies per user in test set
actual_likes = defaultdict(list)
test_df = X_test.copy()
test_df['liked'] = y_test.values
test_df['userId'] = test_df['userId_code'].apply(lambda x: str(user_id_map[x]))
test_df['movieId'] = test_df['movieId_code'].apply(lambda x: str(movie_id_map[x]))
for row in test_df.itertuples():
    if row.liked == 1:
        actual_likes[row.userId].append(row.movieId)

#Evaluate KNN model
print("Evaluating KNN model...")
knn_testset = [(uid, iid, 0) for uid, iid in zip(test_df['userId'], test_df['movieId'])]
knn_predictions = knn_algo.test(knn_testset)
top_k_knn = get_top_k_predictions(knn_predictions, k=5)
knn_prec, knn_rec = precision_recall_at_k(top_k_knn, actual_likes, k=5)

#Evaluate Linear Regression model
print("Evaluating Linear Regression model...")
lr_preds_df = test_df[['userId_code', 'movieId_code', 'userId', 'movieId']].copy()
lr_preds_df['score'] = lr_model.predict(lr_preds_df[['userId_code', 'movieId_code']])

top_k_lr = defaultdict(list)
for row in lr_preds_df.itertuples():
    top_k_lr[row.userId].append((row.movieId, row.score))
for uid in top_k_lr:
    top_k_lr[uid] = sorted(top_k_lr[uid], key=lambda x: x[1], reverse=True)[:5]
lr_prec, lr_rec = precision_recall_at_k(top_k_lr, actual_likes, k=5)


print("\n Evaluation Results for k=5:")
print(f"\n KNN -Precision@5: {knn_prec:.4f}, Recall@5: {knn_rec:.4f}")
print(f"LR - Precision@5: {lr_prec:.4f}, Recall@5: {lr_rec:.4f}")
