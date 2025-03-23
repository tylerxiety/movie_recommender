from flask import Flask, render_template, request, jsonify
import json
import os
import argparse
from model import create_recommender

# Define the function first
def initialize_recommender(use_precomputed=True):
    """Initialize the recommender system"""
    print("Initializing recommender system...")
    recommender = create_recommender(use_precomputed=use_precomputed)
    print("Recommender system ready!")
    return recommender

# Create the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'movie_recommender_secret_key'

# Initialize the recommender globally
recommender = initialize_recommender(use_precomputed=True)

@app.route('/')
def index():
    """Render the home page with recommendations"""
    # Get recommendations
    movie_indices = recommender.get_recommendations(top_k=10)
    recommended_movies = recommender.get_movie_details(movie_indices)
    
    # Get user ratings for initial state
    user_ratings = recommender.user_ratings_df.to_dict('records')
    
    return render_template('index.html', movies=recommended_movies, user_ratings=user_ratings)

@app.route('/rate', methods=['POST'])
def rate_movie():
    """Handle movie rating submissions"""
    data = request.get_json()
    
    imdb_id = data.get('imdb_id', '')
    title = data.get('title', '')
    year = data.get('year', '')
    rating = int(data.get('rating', 0))
    
    # Update rating in the system
    success = recommender.update_rating(imdb_id, title, year, rating)
    
    # Get new recommendations based on updated ratings
    movie_indices = recommender.get_recommendations(top_k=10)
    recommended_movies = recommender.get_movie_details(movie_indices)
    
    # Get all user ratings
    user_ratings = recommender.user_ratings_df.to_dict('records')
    
    return jsonify({
        'success': success,
        'movies': recommended_movies,
        'user_ratings': user_ratings
    })

@app.route('/get_recommendations')
def get_recommendations():
    """Get movie recommendations as JSON"""
    movie_indices = recommender.get_recommendations(top_k=10)
    recommended_movies = recommender.get_movie_details(movie_indices)
    
    # Get all user ratings
    user_ratings = recommender.user_ratings_df.to_dict('records')
    
    return jsonify({
        'movies': recommended_movies,
        'user_ratings': user_ratings
    })

@app.route('/search', methods=['GET'])
def search():
    """Search for movies by query"""
    query = request.args.get('q', '')
    num_results = int(request.args.get('n', 10))
    
    if not query:
        return jsonify({
            'success': False,
            'message': 'No search query provided',
            'movies': []
        })
    
    movie_indices = recommender.search_movies(query, top_k=num_results)
    movie_results = recommender.get_movie_details(movie_indices)
    
    return jsonify({
        'success': True,
        'query': query,
        'movies': movie_results
    })

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Movie Recommender Web App')
    parser.add_argument('--no-precomputed', dest='use_precomputed', action='store_false',
                        help='Do not use precomputed embeddings, compute them at startup')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to run the web server on (default: 5001)')
    parser.set_defaults(use_precomputed=True)
    args = parser.parse_args()
    
    # Re-initialize recommender if command-line args specify not to use precomputed embeddings
    if not args.use_precomputed:
        # We need to modify the global recommender
        # Instead of using global keyword, directly assign to the module-level variable
        globals()['recommender'] = initialize_recommender(use_precomputed=False)
    
    app.run(debug=True, port=args.port) 