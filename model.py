import pandas as pd
import numpy as np
import torch
import faiss
import pickle
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.user_ratings_df = None
        self.index = None
        self.embeddings = None
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_data(self, movies_path, user_ratings_path):
        """Load the movies and user ratings data"""
        print(f"Loading movies data from {movies_path}")
        try:
            # Try with different encoding and error handling options
            self.movies_df = pd.read_csv(
                movies_path, 
                encoding='utf-8',
                on_bad_lines='skip',
                engine='c',  # Use 'c' engine instead of 'python'
                low_memory=False  # Avoid mixed type inference warnings
            )
            # Ensure required columns exist with fallbacks
            required_cols = ['id', 'Name', 'Description', 'PosterLink', 'DatePublished', 'Genres']
            for col in required_cols:
                if col not in self.movies_df.columns:
                    print(f"Warning: Column '{col}' not found. Adding empty column.")
                    self.movies_df[col] = ""
            
            # Fill NaN values with empty strings
            self.movies_df = self.movies_df.fillna('')
            
            # Create a lowercase title column for better matching
            self.movies_df['title_lower'] = self.movies_df['Name'].str.lower()
            
            # Extract year from DatePublished if available
            if 'DatePublished' in self.movies_df.columns:
                self.movies_df['year'] = self.movies_df['DatePublished'].str.extract(r'(\d{4})', expand=False)
            
            print(f"Successfully loaded movies with shape: {self.movies_df.shape}")
        except Exception as e:
            print(f"Error loading movies data: {e}")
            # Create a simple test dataset instead
            test_data = {
                'id': list(range(100)),
                'Name': [f"Movie {i}" for i in range(100)],
                'Description': [f"This is a description for movie {i}" for i in range(100)],
                'PosterLink': ['https://via.placeholder.com/150' for i in range(100)],
                'DatePublished': ['2023-01-01' for i in range(100)],
                'Genres': ['Action, Drama' for i in range(100)],
                'Actors': ['Actor 1, Actor 2' for i in range(100)],
                'Director': ['Director' for i in range(100)]
            }
            self.movies_df = pd.DataFrame(test_data)
            self.movies_df['title_lower'] = self.movies_df['Name'].str.lower()
            self.movies_df['year'] = '2023'
            print("Created test dataset with 100 movies")
            
        print(f"Loading user ratings from {user_ratings_path}")
        try:
            self.user_ratings_df = pd.read_csv(user_ratings_path)
            
            # Create lowercase title column for better matching
            if 'title' in self.user_ratings_df.columns:
                self.user_ratings_df['title_lower'] = self.user_ratings_df['title'].str.lower()
            
            print(f"Successfully loaded {len(self.user_ratings_df)} ratings")
        except Exception as e:
            print(f"Error loading user ratings: {e}")
            # Create empty ratings dataframe
            self.user_ratings_df = pd.DataFrame({
                'imdb_id': [],
                'title': [],
                'title_lower': [],
                'year': [],
                'rating': []
            })
            
        print(f"Loaded {len(self.movies_df)} movies and {len(self.user_ratings_df)} ratings")
    
    def load_model(self):
        """Load the HuggingFace transformer model"""
        print("Loading transformer model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            print(f"Model loaded successfully and running on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding(self, texts, batch_size=32):
        """Get embeddings for a list of texts using batching"""
        if not texts:
            print("Warning: Empty text list provided for embedding")
            return np.array([])
            
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Use tqdm for better progress tracking
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Handle very long texts by truncation
            batch_texts = [text[:4000] if len(text) > 4000 else text for text in batch_texts]
            
            try:
                # Tokenize the texts
                encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                              max_length=512, return_tensors='pt')
                
                # Move to device
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Get model output
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                
                # Mean pooling
                embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                all_embeddings.append(embeddings.cpu().numpy())
            except Exception as e:
                print(f"Error embedding batch starting at index {i}: {e}")
                # Create an empty embedding as fallback for this batch
                embedding_dim = 384  # Dimension for the model being used
                dummy_embeddings = np.zeros((len(batch_texts), embedding_dim))
                all_embeddings.append(dummy_embeddings)
        
        if not all_embeddings:
            print("Warning: Failed to create any embeddings")
            # Return a dummy embedding if all batches failed
            embedding_dim = 384  # Dimension for the model being used
            return np.zeros((len(texts), embedding_dim))
            
        return np.vstack(all_embeddings)
    
    def build_embeddings(self):
        """Build embeddings for all movie descriptions"""
        if not os.path.exists('models'):
            os.makedirs('models')
            
        if os.path.exists('models/movie_embeddings.npy') and os.path.exists('models/faiss_index.bin'):
            print("Loading existing embeddings and index...")
            try:
                self.embeddings = np.load('models/movie_embeddings.npy')
                self.index = faiss.read_index('models/faiss_index.bin')
                print(f"Loaded embeddings with shape: {self.embeddings.shape}")
                return
            except Exception as e:
                print(f"Error loading existing embeddings: {e}")
                print("Will rebuild embeddings...")
        
        # Load model if not already loaded - needed to build embeddings
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        print("Building embeddings for movie descriptions...")
        # Combine name and description for better embeddings
        texts = [f"{name}: {desc}" for name, desc in zip(
            self.movies_df['Name'].tolist(), 
            self.movies_df['Description'].tolist()
        )]
        
        # Process in batches to handle large datasets
        self.embeddings = self.get_embedding(texts)
        
        # Check if we got valid embeddings
        if self.embeddings.size == 0:
            print("Error: Failed to create embeddings")
            return
            
        # Normalize the embeddings
        faiss.normalize_L2(self.embeddings)
        
        # Build the FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        
        # Save the embeddings and index
        try:
            np.save('models/movie_embeddings.npy', self.embeddings)
            faiss.write_index(self.index, 'models/faiss_index.bin')
            print("Successfully saved embeddings and index")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
        
        print(f"Built embeddings with shape: {self.embeddings.shape}")
    
    def load_precomputed_embeddings(self):
        """Load only precomputed embeddings without building new ones"""
        if not os.path.exists('models/movie_embeddings.npy') or not os.path.exists('models/faiss_index.bin'):
            print("Error: Precomputed embeddings not found. Run precompute_embeddings.py first.")
            return False
            
        try:
            print("Loading precomputed embeddings and index...")
            self.embeddings = np.load('models/movie_embeddings.npy')
            self.index = faiss.read_index('models/faiss_index.bin')
            print(f"Loaded embeddings with shape: {self.embeddings.shape}")
            return True
        except Exception as e:
            print(f"Error loading precomputed embeddings: {e}")
            return False
    
    def get_top_rated_movies(self, min_rating=8):
        """Get user's top rated movies"""
        if self.user_ratings_df is None or len(self.user_ratings_df) == 0:
            return pd.DataFrame()
            
        top_rated = self.user_ratings_df[self.user_ratings_df['rating'] >= min_rating]
        return top_rated
    
    def match_movie_to_dataset(self, imdb_id, title, year=None):
        """Match a movie from user ratings to our dataset"""
        # First try to match by IMDB ID if it exists in our dataset
        if 'imdb_id' in self.movies_df.columns:
            matched = self.movies_df[self.movies_df['imdb_id'] == imdb_id]
            if len(matched) > 0:
                return matched.index[0]
        
        # Then try to match by ID (some datasets might use IMDB ID as regular ID)
        if imdb_id and 'id' in self.movies_df.columns:
            # Try with and without 'tt' prefix
            imdb_numeric = imdb_id.replace('tt', '') if isinstance(imdb_id, str) and imdb_id.startswith('tt') else imdb_id
            matched = self.movies_df[self.movies_df['id'].astype(str) == str(imdb_numeric)]
            if len(matched) > 0:
                return matched.index[0]
        
        # Next try exact title match
        title_lower = title.lower() if isinstance(title, str) else ""
        matched = self.movies_df[self.movies_df['title_lower'] == title_lower]
        
        # If we have a year, filter by year too
        if year and len(matched) > 1:
            year_matches = matched[matched['year'] == str(year)]
            if len(year_matches) > 0:
                matched = year_matches
        
        if len(matched) > 0:
            return matched.index[0]
        
        # Try partial title match
        if title_lower:
            matched = self.movies_df[self.movies_df['title_lower'].str.contains(title_lower, na=False)]
            if len(matched) > 0:
                return matched.index[0]
        
        return None
    
    def get_recommendations(self, top_k=10, min_rating=8):
        """Get movie recommendations based on user's top rated movies"""
        # Check if we have a valid index
        if self.index is None or self.embeddings is None or self.embeddings.shape[0] == 0:
            print("Error: No valid index or embeddings found")
            return []
            
        top_rated = self.get_top_rated_movies(min_rating)
        
        if len(top_rated) == 0:
            print("No highly rated movies found, returning popular movies instead.")
            # Return a mix of popular movies
            return self.movies_df.sample(min(top_k, len(self.movies_df))).index.tolist()
        
        # Get the movies from the movies_df based on ratings
        top_rated_movies = []
        for _, row in top_rated.iterrows():
            imdb_id = row.get('imdb_id', '')
            title = row.get('title', '')
            year = row.get('year', None)
            
            movie_idx = self.match_movie_to_dataset(imdb_id, title, year)
            if movie_idx is not None:
                top_rated_movies.append(movie_idx)
        
        if len(top_rated_movies) == 0:
            print("No matches found for rated movies, returning popular movies instead.")
            return self.movies_df.sample(min(top_k, len(self.movies_df))).index.tolist()
            
        # Get the embeddings for the top rated movies
        query_embeddings = self.embeddings[top_rated_movies]
        
        # Compute the average embedding
        avg_embedding = np.mean(query_embeddings, axis=0)
        
        # Normalize the query embedding
        faiss.normalize_L2(avg_embedding.reshape(1, -1))
        
        # Search for similar movies
        distances, indices = self.index.search(avg_embedding.reshape(1, -1), min(top_k + len(top_rated_movies), len(self.movies_df)))
        
        # Filter out the movies that are already in the top rated list
        recs = [idx for idx in indices[0] if idx not in top_rated_movies][:top_k]
        
        return recs
    
    def search_movies(self, query, top_k=10):
        """Search for movies using a text query"""
        if self.index is None or self.embeddings is None or self.embeddings.shape[0] == 0:
            print("Error: No valid index or embeddings found")
            return []
        
        # Get embedding for the query - need to load model if not loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        # Get embedding for the query
        query_embedding = self.get_embedding([query], batch_size=1)
        
        # Normalize the query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search for similar movies
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.movies_df)))
        
        return indices[0].tolist()
    
    def update_rating(self, imdb_id, title, year, rating):
        """Update user ratings with a new rating"""
        try:
            # Check if the movie is already rated
            existing_rating = self.user_ratings_df[self.user_ratings_df['imdb_id'] == imdb_id]
            
            if len(existing_rating) > 0:
                # Update existing rating
                self.user_ratings_df.loc[self.user_ratings_df['imdb_id'] == imdb_id, 'rating'] = rating
            else:
                # Add new rating
                new_rating = pd.DataFrame({
                    'imdb_id': [imdb_id],
                    'title': [title],
                    'year': [year],
                    'rating': [rating]
                })
                # Add lowercase title for matching
                new_rating['title_lower'] = new_rating['title'].str.lower()
                
                self.user_ratings_df = pd.concat([self.user_ratings_df, new_rating], ignore_index=True)
            
            # Save the updated ratings
            self.user_ratings_df.to_csv('data/my_movie_ratings.csv', index=False)
            return True
        except Exception as e:
            print(f"Error updating rating: {e}")
            return False
    
    def get_movie_details(self, movie_indices):
        """Get movie details for the given indices"""
        movies = []
        for idx in movie_indices:
            try:
                if idx < 0 or idx >= len(self.movies_df):
                    print(f"Warning: Index {idx} out of range")
                    continue
                movie = self.movies_df.iloc[idx].to_dict()
                movies.append(movie)
            except Exception as e:
                print(f"Error getting movie details for index {idx}: {e}")
        return movies

# Function to create the recommender
def create_recommender(use_precomputed=True):
    recommender = MovieRecommender()
    recommender.load_data('data/movies_dataset_48k.csv', 'data/my_movie_ratings.csv')
    
    if use_precomputed:
        # Try to load precomputed embeddings
        if recommender.load_precomputed_embeddings():
            print("Successfully loaded precomputed embeddings")
        else:
            # If loading fails, fall back to computing embeddings
            print("Failed to load precomputed embeddings, falling back to computing them")
            recommender.load_model()
            recommender.build_embeddings()
    else:
        # Traditional approach: load model and build embeddings
        recommender.load_model()
        recommender.build_embeddings()
        
    return recommender

if __name__ == "__main__":
    # Test the recommender
    print("Testing the recommender with precomputed embeddings...")
    recommender = create_recommender(use_precomputed=True)
    
    # Test search functionality
    print("\nSearching for 'science fiction space'...")
    search_results = recommender.search_movies("science fiction space", top_k=5)
    search_details = recommender.get_movie_details(search_results)
    for movie in search_details:
        print(f"{movie['Name']} ({movie.get('DatePublished', '').split('-')[0]}) - {movie['Genres']}")
    
    # Test recommendations
    print("\nGetting personalized recommendations...")
    recommendations = recommender.get_recommendations(top_k=10)
    movie_details = recommender.get_movie_details(recommendations)
    
    print("\nRecommended Movies:")
    for movie in movie_details:
        print(f"{movie['Name']} ({movie.get('DatePublished', '').split('-')[0]}) - {movie['Description'][:100]}...") 