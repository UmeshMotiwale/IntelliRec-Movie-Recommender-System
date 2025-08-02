import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

plt.style.use('ggplot')

pd.set_option('display.max_columns',None)

#movies_csv
df = pd.read_csv("D:/movies dataset/ml-32m/movies.csv")
#rating_csv
dff= pd.read_csv("D:/movies dataset/ml-32m/ratings.csv")

combined = pd.merge(df,dff, on='movieId')
combined= combined.drop(['timestamp'],axis=1)

#reducing the data by taking avg of rating for each movie
avg_ratings = combined.groupby('movieId')['rating'].mean().reset_index()

#to see specific columns
avg_ratings.columns = ['movieId', 'avg_rating']

avg_combined= pd.merge(df[['movieId', 'title']],avg_ratings,  on = 'movieId')

avg_combined= avg_combined.drop(['movieId'],axis=1)

#print(avg_combined)

#creating a bar graph of movies and their ratings
avg_combined.head(10).set_index('title').plot(kind='bar', title='Movies and their Ratings')
plt.xlabel("Movie name")
plt.ylabel("Average Rating")
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()

