from flask import Flask, request, render_template_string
import pickle

with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("movie_dict.pkl", "rb") as f:
    movie_dict = pickle.load(f)

app = Flask(__name__)

#Using HTML 
HTML = """
<!doctype html>
<title>Movie Recommender</title>
<h2>IntelliRec: Movie Recommendation System</h2>
<form method=post>
  User ID: <input type=number name=user_id required><br><br>
  Number of recommendations: <input type=number name=num_recs required><br><br>
  <input type=submit value=Recommend>
</form>
{% if recommendations %}
  <h3>Top {{num_recs}} Recommendations for User {{user_id}}:</h3>
  <ul>
  {% for movie, rating in recommendations %}
    <li><b>{{movie}}</b> — predicted rating: {{rating}}</li>
  {% endfor %}
  </ul>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def recommend():
    recommendations = []
    user_id = None
    num_recs = None

    if request.method == "POST":
        user_id = int(request.form["user_id"])
        num_recs = int(request.form["num_recs"])

        movie_ratings = []
        for movieId, title in movie_dict.items():
            try:
                pred = model.predict(user_id, movieId)
                movie_ratings.append((title, pred.est))
            except:
                continue

        top_movies = sorted(movie_ratings, key=lambda x: x[1], reverse=True)[:num_recs]
        recommendations = [(title, round(rating, 2)) for title, rating in top_movies]
    return render_template_string(HTML, recommendations=recommendations, user_id=user_id, num_recs=num_recs)
if __name__ == "__main__":
    app.run(debug=True)
