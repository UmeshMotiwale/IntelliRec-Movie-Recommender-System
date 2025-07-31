import numpy as np
from sklearn.decomposition import TruncatedSVD

# Rows: users, Columns: movies
ratings_matrix = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

N=2 #latent features
svd = TruncatedSVD(N) 
reduced = svd.fit_transform(ratings_matrix)
approx_ratings = np.dot(reduced, svd.components_)


print("\nOriginal Ratings Matrix:")
print(ratings_matrix)

print("\nApproximated Ratings Matrix (after SVD):")
print(np.round(approx_ratings, 2))
