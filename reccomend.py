from flask import Flask, render_template, request, jsonify
import requests
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Replace 'YOUR_TMDB_API_KEY' with your actual TMDB API key
tmdb_api_key = 'YOUR_TMDB_API_KEY'
tmdb_base_url = 'https://api.themoviedb.org/3'
tmdb_movie_search_url = f'{tmdb_base_url}/search/movie'
tmdb_movie_details_url = f'{tmdb_base_url}/movie'

# Preprocess the data and create feature vectors


def preprocess_data(movies_data):
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
        movies_data['tagline'] + ' ' + movies_data['cast'] + \
        ' ' + movies_data['director']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    return similarity

# Function to search for a movie using TMDB API


def search_movie(movie_name):
    params = {
        'api_key': tmdb_api_key,
        'query': movie_name
    }
    response = requests.get(tmdb_movie_search_url, params=params)
    data = response.json()
    if data.get('results'):
        return data['results'][0]
    return None

# Function to get movie details using TMDB API


def get_movie_details(movie_id):
    params = {
        'api_key': tmdb_api_key
    }
    response = requests.get(
        f'{tmdb_movie_details_url}/{movie_id}', params=params)
    data = response.json()
    return data

# Function to recommend movies


def recommend_movies(movie_name, similarity):
    movie = search_movie(movie_name)
    if movie:
        movie_id = movie['id']
        index_of_the_movie = movie_id - 1  # TMDB API IDs start from 1
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(
            similarity_score, key=lambda x: x[1], reverse=True)

        recommendations = []
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data.iloc[index]['title']
            recommendations.append(title_from_index)

            if len(recommendations) >= 5:
                break

        return recommendations
    return []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    movie_name = data['movieName']
    similarity = preprocess_data(movies_data)  # Preprocess data
    recommended_movies = recommend_movies(movie_name, similarity)
    return jsonify(recommended_movies)


if __name__ == '__main__':
    app.run(debug=True)
