import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load your dataset
ratings_csv = "D:/movies dataset/ml-32m/ratings.csv"
movies_csv = "D:/movies dataset/ml-32m/movies.csv"

ratings = pd.read_csv(ratings_csv, nrows=500000)
movies = pd.read_csv(movies_csv)

# Step 1: Binary classification (liked = 1 if rating >= 4)
ratings['liked'] = (ratings['rating'] >= 4).astype(int)

# Step 2: Store original IDs (for decoding later)
ratings['origUserId'] = ratings['userId']
ratings['origMovieId'] = ratings['movieId']

# Step 3: Encode userId and movieId for training
ratings['userId'] = ratings['userId'].astype('category')
ratings['movieId'] = ratings['movieId'].astype('category')
user_id_map = dict(enumerate(ratings['userId'].cat.categories))
movie_id_map = dict(enumerate(ratings['movieId'].cat.categories))
ratings['userId'] = ratings['userId'].cat.codes
ratings['movieId'] = ratings['movieId'].cat.codes

# Step 4: Prepare feature and label sets
X = ratings[['userId', 'movieId']]
y = ratings['liked']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print(f"~ Model trained on {len(ratings)} samples with {len(user_id_map)} users and {len(movie_id_map)} movies.")

# Step 7: Predict on test data
X_test = X_test.copy()
X_test['predicted_score'] = model.predict(X_test)
X_test['true_label'] = y_test.values

# Step 8: Evaluate using Precision@5 and Recall@5
k = 5
precision_scores = []
recall_scores = []

for user_id, group in X_test.groupby('userId'):
    top_k = group.sort_values('predicted_score', ascending=False).head(k)
    true_positives = top_k['true_label'].sum()
    total_actual_positives = group['true_label'].sum()

    precision = true_positives / k if k else 0
    recall = true_positives / total_actual_positives if total_actual_positives else 0

    precision_scores.append(precision)
    recall_scores.append(recall)

# Step 9: Print Evaluation
avg_precision = round(np.mean(precision_scores), 4)
avg_recall = round(np.mean(recall_scores), 4)

print(f"\n!! Evaluation !!:")
print(f"# Precision@{k}: {avg_precision}")
print(f"# Recall@{k}: {avg_recall}")
