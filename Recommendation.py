import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Fill missing data
movies["description"] = movies["description"].fillna("")

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["description"])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def recommend_movie(title):
    if title not in movies["title"].values:
        return "Movie not found!"
    
    idx = movies[movies["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = [movies["title"].iloc[i[0]] for i in sim_scores[1:6]]
    
    return top_movies

# --- User interaction ---
user_input = input("Enter a movie name: ")
recommendations = recommend_movie(user_input)

print("\n🎬 Recommended Movies:")
for movie in recommendations:
    print("-", movie)
