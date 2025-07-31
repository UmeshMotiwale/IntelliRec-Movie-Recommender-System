import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


ratings_csv = "D:/movies dataset/ml-32m/ratings.csv"
movies_csv = "D:/movies dataset/ml-32m/movies.csv"

ratings = pd.read_csv(ratings_csv, nrows=500000)
movies = pd.read_csv(movies_csv)


ratings['origUserId'] = ratings['userId']
ratings['origMovieId'] = ratings['movieId']
ratings['userId'] = ratings['userId'].astype('category')
ratings['movieId'] = ratings['movieId'].astype('category')
user_id_map = dict(enumerate(ratings['userId'].cat.categories))
movie_id_map = dict(enumerate(ratings['movieId'].cat.categories))
ratings['userId'] = ratings['userId'].cat.codes
ratings['movieId'] = ratings['movieId'].cat.codes

#Prepare features and target
X = ratings[['userId', 'movieId']]
y = ratings['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

print(f"\n Trained on {len(ratings)} ratings. Unique users: {len(user_id_map)}, movies: {len(movie_id_map)}.")

#User input
try:
    input_user = int(input(f"\n Enter a userId (real one from dataset): "))
    input_movie = int(input(" Enter a movieId (real one from dataset): "))
    if input_user in ratings['origUserId'].values and input_movie in ratings['origMovieId'].values:
    
        encoded_user = list(user_id_map.keys())[list(user_id_map.values()).index(input_user)]
        encoded_movie = list(movie_id_map.keys())[list(movie_id_map.values()).index(input_movie)]

  
        sample = pd.DataFrame([[encoded_user, encoded_movie]], columns=['userId', 'movieId'])
        predicted_rating = reg_model.predict(sample)[0]

        
        title_row = movies[movies['movieId'] == input_movie]['title']
        movie_title = title_row.values[0] if not title_row.empty else f"Movie ID {input_movie}"

        #Output
        print(f'\n Prediction: User {input_user} is expected to rate "{movie_title}" around {round(predicted_rating, 2)} stars.')
    else:
        print(" User or Movie ID not found in dataset.")

except Exception as e:
    print(" Invalid input.", e)
