# Movie Recommender System

A Flask-based movie recommender system that suggests movies similar to your highly-rated ones based on plot descriptions.

## Features

- Content-based movie recommendations using plot descriptions
- Interactive web interface with movie details (poster, cast, genres, etc.)
- Rating system to provide feedback and improve recommendations
- Real-time recommendation updates based on new ratings

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap, jQuery
- **Machine Learning**: Hugging Face Transformers, FAISS
- **Data Processing**: Pandas, NumPy
- **Model**: all-MiniLM-L6-v2 transformer model for text embeddings

## Project Structure

```
movie_recommender/
├── app.py                 # Main Flask application
├── app_simple.py          # Simplified Flask app
├── minimal_app.py         # Minimal Flask app (currently working version)
├── model.py               # Main recommendation engine
├── model_simple.py        # Simplified recommendation engine
├── data/                  # Data files
│   ├── movies_dataset_48k.csv
│   └── my_movie_ratings.csv
├── models/                # Saved models and embeddings
├── static/                # Static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
└── templates/             # HTML templates
    └── index.html
```

## How It Works

1. Movie descriptions are embedded using Hugging Face Transformers' all-MiniLM-L6-v2 model
2. Embeddings are indexed using FAISS for fast similarity search
3. User's top-rated movies are used to find similar movies based on plot descriptions
4. The system recommends 10 movies with detailed information
5. User can rate movies to update recommendations

## Setup and Installation(Apple Silicon M1)

1. Activate the conda environment:
   ```
   conda activate movrecsys_arm
   ```

2. Install required packages:
   ```
   conda install pytorch
   conda install -c conda-forge faiss-cpu pandas flask tqdm       
   conda install -c pytorch pytorch
   pip install transformers
   ```

3. Run the minimal application (currently working version):
   ```
   python app.py
   ```

4. Open your browser and navigate to: http://127.0.0.1:5001

## Future Improvements

- Better movies dataset
- Advanced recommendation algorithms
- User profiles and personalization
- Deployment to a production server
- Mobile app development

## Troubleshooting

## License

This project is for educational purposes only.

### Updated Workflow with Precomputed Embeddings

The system has been improved to separate the embedding computation process from application startup, making the app start faster:

1. **Precompute Embeddings** (one-time setup):
   ```
   python precompute_embeddings.py
   ```
   This will process the entire dataset and save the embeddings to the `models/` directory.

2. **Run the App** (uses precomputed embeddings by default):
   ```
   python app.py
   ```

3. **Command-line Options**:
   - To force recomputation of embeddings at startup:
     ```
     python app.py --no-precomputed
     ```
   - To specify a different port:
     ```
     python app.py --port 5002
     ```

This separation of concerns makes the application more efficient, as it doesn't need to load the transformer model on every startup when using precomputed embeddings. 